"""Boundary-level verification over recorded episodes.

    python scripts/verify_boundaries.py --episodes artifacts/smoke/openclaw --out artifacts/verify/openclaw
    python scripts/verify_boundaries.py --episodes artifacts/dev/openclaw --out artifacts/verify/compass_v2 \
        --post-method compass_v2 --comp-model gpt-4.1-mini

Without ``--post-method`` the POST arm uses the summary the episode recorded.
With it, the named compressor regenerates a summary from the same absorbable
turns (and the same previous summary), so different compressors are scored on
one shared cohort against a shared PRE control. PRE rollouts are cached under
``<out>/../pre/`` keyed by replay identity and reused across POST methods.
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


def load_episodes(d: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def job(args):
    spec, arm, sample, agent_model, agent_provider, post_method, comp_model, comp_provider, budget, tag = args
    from compass.eval.replay import rollout
    from compass.harness.llm import LLM
    agent = LLM(agent_model, agent_provider, max_tokens=2048)
    summary, render, extra, setup = None, None, None, None
    if arm == "POST" and post_method:
        from run_episodes import make_compressor
        comp = make_compressor(post_method, comp_model, comp_provider, budget)
        if post_method.startswith("compass") and spec["prev_summary"] is not None:
            # a graph compressor keeps state across boundaries; on a cohort recorded under
            # another compressor it gets the whole prefix instead of a foreign summary
            summary = comp.compress(spec["instruction"], None, spec["prefix_turns"])
        else:
            summary = comp.compress(spec["instruction"], spec["prev_summary"], spec["absorbable_turns"])
        render = comp.render_context
        extra = getattr(comp, "last_extra", None)
        setup = getattr(comp, "last_setup_code", None)
    r = rollout(spec, arm, agent, summary=summary, render_context=render, sample=sample,
                experiment_name=f"{tag}_{spec['replay_identity']}", setup_code=setup)
    r["post_method"] = post_method if arm == "POST" else None
    r["extra"] = extra
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="dir of episode JSONs (one method)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--post-method", default=None)
    ap.add_argument("--agent-model", default="deepseek-v4-flash:0731")
    ap.add_argument("--agent-provider", default="ollama")
    ap.add_argument("--comp-model", default="deepseek-v4-flash:0731")
    ap.add_argument("--comp-provider", default="ollama")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--n-first", type=int, default=3)
    ap.add_argument("--n-later", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-boundaries", type=int, default=0)
    ap.add_argument("--tag", default="v")
    a = ap.parse_args()

    from compass.eval.burden import format_table, scan
    from compass.eval.replay import boundary_specs

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre_path = out.parent / "pre" / "rollouts.jsonl"
    pre_path.parent.mkdir(parents=True, exist_ok=True)
    post_path = out / "rollouts.jsonl"

    specs = []
    for ep in load_episodes(Path(a.episodes)):
        for s in boundary_specs(ep):
            s["instruction"] = ep["instruction"]
            specs.append(s)
    if a.max_boundaries:
        specs = specs[:a.max_boundaries]
    (out / "boundaries.json").write_text(json.dumps(
        [{k: v for k, v in s.items() if k not in ("pre", "post", "absorbable_turns", "prefix_codes")}
         for s in specs], indent=1), encoding="utf-8")
    print(f"{len(specs)} boundaries from {a.episodes}")

    pre_done = {(r["replay_identity"], r["sample"]) for r in read_jsonl(pre_path)}
    post_done = {(r["boundary_id"], r["sample"]) for r in read_jsonl(post_path)}
    jobs = []
    for s in specs:
        n = a.n_first if s["first"] else a.n_later
        for i in range(n):
            if (s["replay_identity"], i) not in pre_done:
                jobs.append((s, "PRE", i, a.agent_model, a.agent_provider, None, a.comp_model, a.comp_provider, a.budget, a.tag))
            if (s["boundary_id"], i) not in post_done:
                jobs.append((s, "POST", i, a.agent_model, a.agent_provider, a.post_method, a.comp_model, a.comp_provider, a.budget, a.tag))
    print(f"{len(jobs)} rollouts to run")

    def sink(r):
        p = pre_path if r["arm"] == "PRE" else post_path
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {r['boundary_id']} {r['arm']}#{r['sample']} steps={len(r['steps'])} "
              f"E={sum(s['is_error'] for s in r['steps'])} R={sum(s['refetch'] for s in r['steps'])} "
              f"{r['termination']} {r.get('error') or ''}", flush=True)

    if a.workers <= 1:
        for j in jobs:
            sink(job(j))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for f in as_completed([ex.submit(job, j) for j in jobs]):
                sink(f.result())

    ident = {s["replay_identity"]: s["boundary_id"] for s in specs}
    pre = [dict(r, boundary_id=ident[r["replay_identity"]]) for r in read_jsonl(pre_path) if r["replay_identity"] in ident]
    post = [r for r in read_jsonl(post_path) if r["boundary_id"] in ident.values()]
    res = scan(pre + post)
    (out / "scan.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nPOST={a.post_method or 'recorded'}  boundaries={len(res['per_boundary'])}")
    print(format_table(res["table"]))


if __name__ == "__main__":
    main()
