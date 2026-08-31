"""OfficeBench (Wang et al., 2024) episodes with the same compaction loop as AppWorld.

The agent is ACON's OfficeBench agent: JSON tool actions (``{"app": ..,
"action": .., ...}``) inside ``<think>..</think><action>..</action>``, one action
per turn, apps switched explicitly. This is a ReAct/tool-calling agent, not a
code-acting one, so it exercises the schema-aware adapter tier of COMPASS.

Runs on Linux (WSL): the environment shells out to ``python3`` app scripts and
LibreOffice. Each episode copies the task directory into its own working
directory and chdirs there, because the environment resolves paths relative
to the process working directory.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from ..eval.signatures import canon
from .compressors import Compressor
from .episode import OBS_STORE_LIMIT, build_messages
from .llm import LLM
from .prompt import count_tokens, turns_to_text

ERROR_PREFIXES = ("Error", "Command failed", "Command timed out", "Malformed")

PROMPT_UNDECIDED = ("##Available apps: {available_apps}\n##Instruction:\n"
                    " - choose an app from the available apps: "
                    "{{\"app\": \"system\", \"action\": \"switch_app\", \"target_app\": [THE_APP_YOU_CHOOSE]}}\n##Command:")
PROMPT_DECIDED = ("##Current apps: {current_app}\n##Instruction: Choose one action from the list as the next step. "
                  "Use the JSON schema provided to format your response. You may optionally include your thinking process.\n"
                  "{detailed_instruction} - switch to another app among {available_apps}: "
                  "{{\"app\": \"system\", \"action\": \"switch_app\", \"target_app\": [THE_APP_YOU_CHOOSE] }}\n"
                  " - finish the task with your answer as None if the task is not a question: "
                  "<think>I'm finished the task.</think><action>{{\"app\": \"system\", \"action\": \"finish_task\", \"answer\": \"None\"}}</action>\n"
                  " - finish the task with your answer if the task is a question: "
                  "<think>I'm finished and the answer is [answer]</think><action>{{\"app\": \"system\", \"action\": \"finish_task\", \"answer\": [ANSWER]}}</action>\n"
                  "##Command:")


def is_ob_error(observation: str | None) -> bool:
    o = (observation or "").strip()
    return o.startswith(ERROR_PREFIXES)


def ob_signature(action_json: str) -> tuple[str | None, dict | None]:
    """``app.action(k=v,...)`` canonical signature of an OfficeBench action, and the parsed dict."""
    try:
        a = json.loads(action_json)
    except Exception:  # noqa: BLE001
        return None, None
    if not isinstance(a, dict) or "action" not in a:
        return None, None
    app = a.get("app", "system")
    act = str(a["action"])
    if "." in act:
        app, act = act.split(".", 1)
    if act == "switch_app":
        app = "system"
    args = ",".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in sorted(a.items()) if k not in ("app", "action"))
    return f"{app}.{act}({canon(args)})", a


def extract_action(response: str) -> str:
    """ACON's processor: the JSON object between the first '{' and the last '}'."""
    if "{" not in response or "}" not in response:
        return response.strip()
    return response[response.find("{"): response.rfind("}") + 1]


def think_of(response: str) -> str:
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return m.group(1).strip() if m else ""


def load_prompts(acon_root: Path) -> dict:
    return json.loads((acon_root / "experiments/officebench/prompts/prompts_v2.json").read_text(encoding="utf-8"))


def _detailed_instruction(env) -> str:
    from productive_agents.env.officebench import apps
    out = ""
    for action in env.get_available_actions():
        mod = apps.AVAILABLE_ACTIONS.get(env.current_app, {}).get(action)
        out += f" - {getattr(mod, 'DEMO', action + ': (no description available)')}\n"
    return out


def _postfix(env) -> str:
    apps_ = list(env.available_apps.keys())
    if env.current_app is None:
        return PROMPT_UNDECIDED.format(available_apps=apps_)
    return PROMPT_DECIDED.format(current_app=env.current_app, detailed_instruction=_detailed_instruction(env),
                                 available_apps=[a for a in apps_ if a != env.current_app])


def evaluate_testbed(task_config: dict, testbed_dir: str) -> bool:
    from productive_agents.env.officebench import evaluate as ev
    for item in task_config["evaluation"]:
        fn = getattr(ev, item["function"], None)
        if fn is None or not fn(testbed_dir, item["args"]):
            return False
    return True


def run_officebench_episode(task_id: str, subtask_id: str, agent: LLM, compressor: Compressor | None, *,
                            acon_root: str | Path, work_root: str | Path, budget: int = 4096, max_steps: int = 30,
                            preserve_last_k: int = 1, out_path: str | Path | None = None) -> dict:
    from productive_agents.env.officebench import OfficeBenchEnv, OfficeBenchEnvConfig

    acon_root = Path(acon_root).resolve()
    t0 = time.time()
    method = compressor.name if compressor else "full"
    run_dir = Path(work_root).resolve() / method / f"{task_id}_{subtask_id}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    task_dir_rel = f"tasks/{task_id}"
    shutil.copytree(acon_root / "experiments/officebench/tasks" / task_id, run_dir / task_dir_rel)
    task_config = json.loads((run_dir / task_dir_rel / "subtasks" / f"{subtask_id}.json").read_text(encoding="utf-8"))
    task_config["testbed_data_path"] = f"{task_dir_rel}/testbed/data"
    prompts = load_prompts(acon_root)

    rec = {"task_id": f"{task_id}/{subtask_id}", "env": "officebench", "method": method, "agent_model": agent.model,
           "budget": budget, "max_steps": max_steps, "obs_limit": OBS_STORE_LIMIT, "instruction": task_config["task"],
           "steps": [], "boundaries": [], "termination_reason": None, "error": None}
    turns: list[dict] = []
    summary: str | None = None
    absorbed_upto = 0
    peak = 0
    seen: set[str] = set()
    window: list[str] = []
    old_cwd = os.getcwd()
    os.chdir(run_dir)
    env = None
    try:
        cfg = OfficeBenchEnvConfig(local_workdir=".", task_dir=task_dir_rel, task=task_config["task"])
        env = OfficeBenchEnv(config=cfg)
        env.reset()
        intro = "".join(f" - {m.INTRO}\n" for m in env.available_apps.values() if hasattr(m, "INTRO"))
        system = prompts["system_message"].format_map({
            "username": task_config.get("username", "user"), "date": task_config.get("date", ""),
            "weekday": task_config.get("weekday", ""), "time": task_config.get("time", ""),
            "app_introduction": intro, "testbed_data_path": task_config["testbed_data_path"]}).strip()
        first = prompts["prompt_undecided_app"].format_map({"task": task_config["task"],
                                                            "available_apps": list(env.available_apps.keys())})
        reason = "max_steps"
        for step in range(1, max_steps + 1):
            if compressor is not None:
                absorbable = turns[absorbed_upto: max(absorbed_upto, len(turns) - preserve_last_k)]
                if absorbable:
                    load = count_tokens((summary or "") + turns_to_text(
                        [{**t, "observation": t["observation"][:OBS_STORE_LIMIT]} for t in absorbable]))
                    if load > budget:
                        prev, tb = summary, time.time()
                        try:
                            summary = compressor.compress(task_config["task"], prev, absorbable)
                            status = "ok"
                        except Exception as e:  # noqa: BLE001
                            status, summary = f"compressor_error: {e}", prev
                        rec["boundaries"].append({
                            "before_step": step, "absorbed_steps": [absorbable[0]["step"], absorbable[-1]["step"]],
                            "retained_steps": [t["step"] for t in turns[len(turns) - preserve_last_k:]],
                            "load_tokens": load, "prev_summary": prev, "summary": summary,
                            "summary_tokens": count_tokens(summary or ""), "status": status,
                            "wall": round(time.time() - tb, 2), "extra": getattr(compressor, "last_extra", None)})
                        if status == "ok":
                            absorbed_upto = len(turns) - preserve_last_k
            block = compressor.render_context(summary) if (compressor is not None and summary) else None
            raw_turns = [{"code": t["code"], "observation": t["prompt_obs"]} for t in turns[absorbed_upto:]]
            msgs = [{"role": "system", "content": system}] + build_messages(first + (block or ""), None, raw_turns)[1:]
            ptok = sum(count_tokens(m["content"]) for m in msgs)
            peak = max(peak, ptok)
            try:
                raw = agent.chat(msgs, tag="agent")
            except Exception as e:  # noqa: BLE001
                reason, rec["error"] = "agent_error", str(e)
                break
            action = extract_action(raw)
            window = (window + [action])[-5:]
            if len(window) == 5 and all(w == window[0] for w in window):
                action = json.dumps({"app": "system", "action": "got_stuck"})
            obs, reward, done, info = env.step(action)
            obs = obs or ""
            sig, parsed = ob_signature(action)
            refetch = sig is not None and sig in seen and (parsed or {}).get("action") != "switch_app"
            if sig:
                seen.add(sig)
            prompt_obs = obs[:OBS_STORE_LIMIT] + "\n" + _postfix(env)
            turns.append({"step": step, "code": action, "observation": obs, "prompt_obs": prompt_obs,
                          "reasoning": think_of(raw), "env": "officebench"})
            rec["steps"].append({"step": step, "code": action, "raw_response": raw[:3000],
                                 "observation": obs[:OBS_STORE_LIMIT], "is_error": is_ob_error(obs),
                                 "sigs": [sig] if sig else [], "refetch": refetch, "prompt_tokens": ptok,
                                 "compressed": summary is not None, "current_app": env.current_app})
            if done:
                reason = "task_completed" if reward > 0 else "gave_up"
                break
        rec["termination_reason"] = reason
        ok = evaluate_testbed(task_config, str(run_dir / task_dir_rel / "testbed"))
        rec["success"], rec["score"] = bool(ok), 1.0 if ok else 0.0
    except Exception as e:  # noqa: BLE001
        rec["termination_reason"], rec["error"] = "harness_error", f"{type(e).__name__}: {e}"
        rec.setdefault("success", False)
        rec.setdefault("score", 0.0)
    finally:
        os.chdir(old_cwd)
        try:
            if env is not None:
                env.close()
        except Exception:  # noqa: BLE001
            pass
    rec["n_steps"] = len(rec["steps"])
    rec["n_blocked"] = sum(s["is_error"] for s in rec["steps"])
    rec["n_refetch"] = sum(s["refetch"] for s in rec["steps"])
    rec["peak_prompt_tokens"] = peak
    rec["agent_usage"] = agent.usage.to_dict()
    rec["compressor_usage"] = compressor.llm.usage.to_dict() if (compressor and compressor.llm) else None
    rec["wall_seconds"] = round(time.time() - t0, 1)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec
