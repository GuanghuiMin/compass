"""Phase 1 offline graph-construction study (no environment needed).

Corpus: the six full-context trajectories and the five packet episodes shipped
in the collaborator's v1 repo, plus any episode JSONs under --extra.

For every episode: build the deterministic graph step by step, simulate a
4096-token compaction schedule (preserve last turn), and at each boundary run
the COMPASS compressor. Report:
  build_ok               deterministic build never fails
  consumes_resolution    fraction of free variable loads resolved to a producer
  needs P/R/F1           predicted `needs` of open intents vs the variables
                         actually consumed by steps AFTER the boundary
  live_recall            fraction of later-consumed variables kept LIVE
  level / tokens / drops fold level, handover size, dropped proposals
  done_grounded          done intents whose evidence includes a non-doc API call

    python scripts/offline_graph_study.py --out reports/phase1_offline.json [--no-llm]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
V1 = ROOT.parent / "compass_v1"


def load_corpus(extra: list[str]) -> list[dict]:
    eps = []
    for p in sorted((V1 / "artifacts/diagnostic/six_task_v1/full_context").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            steps = [{"step": s["step_id"], "code": s.get("code") or "", "observation": s.get("observation") or "",
                      "reasoning": s.get("reasoning") or ""} for s in r["trajectory_steps"]]
            goal = r.get("instruction")
            if not goal:
                try:
                    from compass import compat
                    compat.install()
                    from appworld.task import Task
                    goal = Task.load(task_id=r["task_id"]).instruction
                except Exception:  # noqa: BLE001
                    goal = None
            eps.append({"id": f"v1full::{r['task_id']}", "goal": goal, "steps": steps})
    best: dict[str, dict] = {}
    for f in glob.glob(str(V1 / "inputs/diagnostic/packets/*.json")):
        p = json.loads(Path(f).read_text(encoding="utf-8"))
        if len(p["prefix"]) > len(best.get(p["goal"], {"prefix": []})["prefix"]):
            best[p["goal"]] = p
    for i, p in enumerate(best.values()):
        steps = [{"step": j + 1, "code": s.get("code") or "", "observation": s.get("observation") or "",
                  "reasoning": s.get("reasoning") or ""} for j, s in enumerate(p["prefix"])]
        eps.append({"id": f"v1packet::{i}", "goal": p["goal"], "steps": steps})
    for d in extra:
        for f in sorted(Path(d).glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            eps.append({"id": f"ep::{r['method']}::{r['task_id']}", "goal": r["instruction"],
                        "steps": [{"step": s["step"], "code": s["code"], "observation": s["observation"],
                                   "reasoning": ""} for s in r["steps"]]})
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/phase1_offline.json")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--comp-model", default="gpt-4.1-mini")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from compass.graph.build import Graph
    from compass.graph.compressor import CompassCompressor
    from compass.harness.compressors import clip_turns
    from compass.harness.llm import LLM
    from compass.harness.prompt import count_tokens, turns_to_text

    eps = load_corpus(a.extra)
    if a.limit:
        eps = eps[:a.limit]
    llm = None if a.no_llm else LLM(a.comp_model, "openai", max_tokens=4096)
    report = {"episodes": [], "summary": {}}
    for ep in eps:
        steps = ep["steps"]
        goal = ep["goal"] or ep["id"]
        g_full = Graph(goal)
        n_uses = n_res = 0
        for t in clip_turns(steps):
            s = g_full.add_turn(t)
            n_uses += len(s["uses"])
            n_res += len(s["consumes"])
        comp = CompassCompressor(llm, a.budget, use_llm=not a.no_llm)
        prev, absorbed = None, 0
        rec = {"id": ep["id"], "n_steps": len(steps), "consumes_resolution": round(n_res / max(n_uses, 1), 3),
               "n_uses": n_uses, "boundaries": []}
        for k in range(2, len(steps) + 1):
            absorbable = steps[absorbed:k - 1]          # everything but the preserved last turn
            if not absorbable:
                continue
            load = count_tokens((prev or "") + turns_to_text(clip_turns(absorbable)))
            if load <= a.budget:
                continue
            try:
                text = comp.compress(goal, prev, absorbable)
            except Exception as e:  # noqa: BLE001
                rec["boundaries"].append({"before_step": k, "ok": False, "error": str(e)[:200]})
                continue
            ex = comp.last_extra or {}
            g = Graph.from_dict(comp._graphs[list(comp._graphs)[-1]])
            # GT: infos produced before the boundary that a step at/after it reads
            future_names = set()
            for s in steps[k - 1:]:
                st = g_full.steps.get(f"s{s['step']}")
                if st:
                    for c in st["consumes"]:
                        info = g_full.infos[c]
                        if int(info.producer[1:]) < k:
                            future_names.add(info.name)
            open_needs = set()
            for it in g.intents.values():
                if it.status not in ("done", "invalidated"):
                    open_needs.update(g.infos[n].name for n in it.needs if n in g.infos)
            live_names = {i.name for i in g.live_infos() if i.kind == "runtime_reference"}
            tp = len(open_needs & future_names)
            P = tp / len(open_needs) if open_needs else None
            R = tp / len(future_names) if future_names else None
            F = (2 * P * R / (P + R)) if (P and R) else (0.0 if (open_needs or future_names) else None)
            live_R = len(live_names & future_names) / len(future_names) if future_names else None
            dones = [it for it in g.intents.values() if it.status == "done"]
            dg = sum(1 for it in dones if any(
                any(not api.startswith("api_docs") for api in g.steps[e]["api_names"])
                for e in it.evidence if e in g.steps))
            rec["boundaries"].append({
                "before_step": k, "ok": True, "level": ex.get("level"), "degraded": ex.get("degraded"),
                "tokens": ex.get("summary_tokens"), "n_intents": ex.get("n_intents"), "n_infos": ex.get("n_infos"),
                "drops": len(ex.get("drops") or []), "refine": ex.get("refine"),
                "needs_pred": sorted(open_needs), "needs_gt": sorted(future_names),
                "needs_P": P, "needs_R": R, "needs_F1": F, "live_recall": live_R,
                "done_total": len(dones), "done_grounded": dg, "handover": text,
                "drop_reasons": [d.get("drop") for d in g.log],
            })
            prev, absorbed = text, k - 1
        report["episodes"].append(rec)
        bs = rec["boundaries"]
        f1s = [b["needs_F1"] for b in bs if b.get("needs_F1") is not None]
        print(f"{ep['id']:34} steps={len(steps):3} cons_res={rec['consumes_resolution']:.2f} "
              f"bnd={len(bs)} lvl={[b.get('level') for b in bs]} tok={[b.get('tokens') for b in bs]} "
              f"F1={[round(x, 2) for x in f1s]} liveR={[round(b['live_recall'], 2) for b in bs if b.get('live_recall') is not None]} "
              f"drops={[b.get('drops') for b in bs]}", flush=True)

    allb = [b for e in report["episodes"] for b in e["boundaries"] if b.get("ok")]
    nb_all = sum(len(e["boundaries"]) for e in report["episodes"])
    f1s = [b["needs_F1"] for b in allb if b.get("needs_F1") is not None]
    lr = [b["live_recall"] for b in allb if b.get("live_recall") is not None]
    report["summary"] = {
        "n_episodes": len(report["episodes"]), "n_boundaries": len(allb),
        "build_ok_rate": len(allb) / max(1, nb_all),
        "mean_consumes_resolution": sum(e["consumes_resolution"] for e in report["episodes"]) / max(1, len(report["episodes"])),
        "within_budget_rate": sum(1 for b in allb if b["level"] in (0, 1, 2)) / max(1, len(allb)),
        "level_hist": {str(l): sum(1 for b in allb if b["level"] == l) for l in (0, 1, 2, 3)},
        "mean_tokens": sum(b["tokens"] or 0 for b in allb) / max(1, len(allb)),
        "mean_needs_F1": sum(f1s) / max(1, len(f1s)), "n_F1": len(f1s),
        "mean_live_recall": sum(lr) / max(1, len(lr)),
        "total_drops": sum(b["drops"] or 0 for b in allb),
        "drop_reasons": dict(__import__("collections").Counter(
            r for e in report["episodes"] for b in e["boundaries"][-1:] for r in b.get("drop_reasons", []))),
        "done_grounded_rate": sum(b["done_grounded"] for b in allb) / max(1, sum(b["done_total"] for b in allb)),
        "llm_usage": llm.usage.to_dict() if llm else None,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
