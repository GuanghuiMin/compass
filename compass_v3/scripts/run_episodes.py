"""Run AppWorld episodes for one method over a task list.

    python scripts/run_episodes.py --method openclaw --tasks test_normal:0:3 --out artifacts/smoke
    python scripts/run_episodes.py --method full --tasks 3d9a636_1,fd1f8fa_2 --out artifacts/smoke

Each episode is written to ``<out>/<method>/<task_id>.json`` and skipped if the
file already exists (resumable). ``--workers N`` runs N processes.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def resolve_tasks(spec: str) -> list[str]:
    from appworld import load_task_ids
    out = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            split, lo, hi = part.split(":")
            out.extend(load_task_ids(split)[int(lo):int(hi)])
        elif part in ("test_normal", "test_challenge", "train", "dev"):
            out.extend(load_task_ids(part))
        else:
            out.append(part)
    return out


def make_compressor(method: str, model: str, provider: str, budget: int):
    from compass.harness.llm import LLM
    if method == "full":
        return None
    if method == "fifo":
        from compass.harness.compressors import FIFO
        return FIFO(None, budget)
    from compass.harness.compressors import BASELINES
    if method in BASELINES:
        return BASELINES[method](LLM(model, provider, max_tokens=4096), budget)
    if method.startswith("compass"):
        from compass.graph.compressor import make_compass
        return make_compass(method, LLM(model, provider, max_tokens=4096), budget)
    raise SystemExit(f"unknown method {method}")


def one(args_tuple):
    (task_id, method, agent_model, agent_provider, comp_model, comp_provider, budget,
     max_steps, out_dir, run_tag, agent_temp) = args_tuple
    from compass.harness.episode import run_episode
    from compass.harness.llm import LLM
    out_path = Path(out_dir) / method / f"{task_id}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    agent = LLM(agent_model, agent_provider, max_tokens=2048, temperature=agent_temp)
    comp = make_compressor(method, comp_model, comp_provider, budget)
    return run_episode(task_id, agent, comp, budget=budget, max_steps=max_steps,
                       experiment_name=f"{run_tag}_{method}_{task_id}", out_path=out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    # defaults follow the endpoint policy: development runs on Ollama Cloud;
    # pass --agent-provider openai --agent-model gpt-4.1 only for paper runs
    ap.add_argument("--agent-model", default="deepseek-v4-flash:0731")
    ap.add_argument("--agent-provider", default="ollama")
    ap.add_argument("--comp-model", default="deepseek-v4-flash:0731")
    ap.add_argument("--comp-provider", default="ollama")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--agent-temp", type=float, default=0.0)
    a = ap.parse_args()
    tasks = resolve_tasks(a.tasks)
    jobs = [(t, a.method, a.agent_model, a.agent_provider, a.comp_model, a.comp_provider,
             a.budget, a.max_steps, a.out, a.tag, a.agent_temp) for t in tasks]
    results = []

    def report(r):
        print(f"{r['task_id']}  success={r.get('success')} score={r.get('score', 0):.2f} steps={r['n_steps']} "
              f"bnd={len(r['boundaries'])} blocked={r['n_blocked']} refetch={r['n_refetch']} "
              f"peak={r['peak_prompt_tokens']} {r['termination_reason']} {r.get('error') or ''}", flush=True)

    if a.workers <= 1:
        for j in jobs:
            r = one(j)
            results.append(r)
            report(r)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(one, j): j[0] for j in jobs}
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                report(r)
    n = len(results)
    acc = sum(bool(r.get("success", False)) for r in results) / max(n, 1)
    print(f"\n{a.method}: {n} tasks  acc={acc:.3f}  mean_steps={sum(r['n_steps'] for r in results)/max(n,1):.1f}  "
          f"agent_prompt_tokens={sum(r['agent_usage']['prompt_tokens'] for r in results)}")


if __name__ == "__main__":
    main()
