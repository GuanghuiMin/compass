"""Run OfficeBench episodes (Linux/WSL) for one method.

    python scripts/run_officebench.py --method openclaw --split test --out artifacts/ob --workers 2
    python scripts/run_officebench.py --method full --tasks 1-13/0,1-14/0 --out artifacts/ob

``--split test`` uses ACON's test_tasks.txt restricted to non-image tasks; every
subtask of each task is an episode. Episodes are written to
``<out>/<method>/<task>_<sub>.json`` and skipped if present.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
ACON = ROOT.parent / "acon_ref"
OB = ACON / "experiments/officebench"


def resolve(spec: str | None, split: str | None) -> list[tuple[str, str]]:
    if spec:
        return [tuple(p.strip().split("/")) for p in spec.split(",")]
    ids = [l.strip() for l in (OB / f"{split}_tasks.txt").read_text().splitlines() if l.strip()]
    non_image = {l.strip() for l in (OB / "non_image_tasks.txt").read_text().splitlines() if l.strip()}
    out = []
    for t in ids:
        if t not in non_image:
            continue
        for f in sorted((OB / "tasks" / t / "subtasks").glob("*.json")):
            out.append((t, f.stem))
    return out


def one(args):
    (task, sub, method, agent_model, agent_provider, comp_model, comp_provider, budget, max_steps,
     out_dir, work_root, temp) = args
    from compass.harness.llm import LLM
    from compass.harness.officebench import run_officebench_episode
    from run_episodes import make_compressor
    out_path = Path(out_dir) / method / f"{task}_{sub}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    agent = LLM(agent_model, agent_provider, max_tokens=2048, temperature=temp)
    comp = make_compressor(method, comp_model, comp_provider, budget)
    return run_officebench_episode(task, sub, agent, comp, acon_root=ACON, work_root=work_root, budget=budget,
                                   max_steps=max_steps, out_path=out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--agent-model", default="deepseek-v4-flash:0731")
    ap.add_argument("--agent-provider", default="ollama")
    ap.add_argument("--comp-model", default="deepseek-v4-flash:0731")
    ap.add_argument("--comp-provider", default="ollama")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--work-root", default=None)
    ap.add_argument("--agent-temp", type=float, default=0.0)
    a = ap.parse_args()
    work_root = a.work_root or str(Path(a.out) / "_work")
    jobs = [(t, s, a.method, a.agent_model, a.agent_provider, a.comp_model, a.comp_provider, a.budget, a.max_steps,
             a.out, work_root, a.agent_temp) for t, s in resolve(a.tasks, a.split)]
    print(f"{len(jobs)} episodes", flush=True)
    results = []

    def report(r):
        print(f"{r['task_id']:10} success={r.get('success')} steps={r['n_steps']} bnd={len(r['boundaries'])} "
              f"blocked={r['n_blocked']} refetch={r['n_refetch']} peak={r['peak_prompt_tokens']} "
              f"{r['termination_reason']} {r.get('error') or ''}", flush=True)

    if a.workers <= 1:
        for j in jobs:
            r = one(j)
            results.append(r)
            report(r)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for f in as_completed([ex.submit(one, j) for j in jobs]):
                r = f.result()
                results.append(r)
                report(r)
    n = len(results)
    print(f"\n{a.method}: {n} episodes  acc={sum(bool(r.get('success')) for r in results)/max(n,1):.3f}  "
          f"mean_steps={sum(r['n_steps'] for r in results)/max(n,1):.1f}")


if __name__ == "__main__":
    main()
