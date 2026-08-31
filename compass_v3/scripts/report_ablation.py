"""Paired ablation report beyond accuracy: missed requirements, duplicated side effects,
premature completion, post-compaction burden, graph/context size.

Usage: python scripts/report_ablation.py <runs_dir> <method> [<method> ...]
where <runs_dir>/<method>/*.json are episode records from run_episodes.py.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

MUTATING = re.compile(r"^(create|send|add|delete|remove|update|pay|transfer|post|move|complete|like|approve|deny)_?")
ANSWER_RE = re.compile(r"complete_task\((.*)\)", re.S)


def episode_metrics(e: dict) -> dict:
    steps = e.get("steps", [])
    bounds = e.get("boundaries", [])
    # duplicated side effects: the same successful mutating call signature executed more than once
    seen: dict[str, int] = {}
    for s in steps:
        sig = s.get("exact") or ""
        name = sig.split("(", 1)[0].split(".")[-1]
        if sig and MUTATING.match(name) and "Execution failed" not in (s.get("observation") or ""):
            seen[sig] = seen.get(sig, 0) + 1
    dup = sum(c - 1 for c in seen.values() if c > 1)
    last = steps[-1]["code"] if steps else ""
    m = ANSWER_RE.search(last)
    completed = bool(m)
    spurious_answer = bool(m and "answer" in (m.group(1) or "")) and not e.get("success")
    premature = completed and not e.get("success") and float(e.get("score") or 0) >= 0.5
    # open requirements at the moment of completion (frontier variants record the frontier size)
    missed = None
    for b in reversed(bounds):
        x = b.get("extra") or {}
        if x.get("frontier") is not None:
            missed = x.get("frontier") if completed else None
            break
    gb = [x.get("graph_bytes") for b in bounds for x in [b.get("extra") or {}] if x.get("graph_bytes")]
    stok = [b.get("summary_tokens") for b in bounds if b.get("summary_tokens")]
    return {"success": bool(e.get("success")), "steps": len(steps), "maxed": len(steps) >= 50,
            "boundaries": len(bounds), "dup_side_effects": dup, "premature": premature,
            "spurious_answer": spurious_answer, "refetch": e.get("n_refetch") or 0,
            "blocked": e.get("n_blocked") or 0, "open_reqs_at_completion": missed,
            "graph_bytes": st.mean(gb) if gb else 0, "summary_tokens": st.mean(stok) if stok else 0}


def main() -> None:
    root = Path(sys.argv[1])
    methods = sys.argv[2:]
    cols = ["acc", "n", "maxed", "premature", "spurious_ans", "dup_eff/ep", "refetch/ep",
            "blocked/ep", "open_req@done", "bnd/ep", "graph_kB", "sum_tok"]
    print(f"{'method':26s} " + " ".join(f"{c:>13s}" for c in cols))
    base: dict[str, dict] | None = None
    base_name = methods[0] if methods else ""
    for m in methods:
        files = sorted((root / m).glob("*.json"))
        if not files:
            print(f"{m:26s}  (no episodes)")
            continue
        eps = {f.stem: episode_metrics(json.loads(f.read_text(encoding="utf-8"))) for f in files}
        v = list(eps.values())
        openr = [x["open_reqs_at_completion"] for x in v if x["open_reqs_at_completion"] is not None]
        row = [f"{st.mean(x['success'] for x in v):.3f}", f"{len(v)}", f"{sum(x['maxed'] for x in v)}",
               f"{sum(x['premature'] for x in v)}", f"{sum(x['spurious_answer'] for x in v)}",
               f"{st.mean(x['dup_side_effects'] for x in v):.2f}", f"{st.mean(x['refetch'] for x in v):.1f}",
               f"{st.mean(x['blocked'] for x in v):.1f}",
               (f"{st.mean(openr):.2f}" if openr else "-"),
               f"{st.mean(x['boundaries'] for x in v):.1f}", f"{st.mean(x['graph_bytes'] for x in v)/1000:.1f}",
               f"{st.mean(x['summary_tokens'] for x in v):.0f}"]
        print(f"{m:26s} " + " ".join(f"{c:>13s}" for c in row))
        if base is None:
            base = eps
        else:
            shared = [t for t in eps if t in base]
            if shared:
                d = st.mean(eps[t]["success"] for t in shared) - st.mean(base[t]["success"] for t in shared)
                print(f"{'':26s}   paired acc vs {base_name}: {d:+.3f} on {len(shared)} shared tasks")


if __name__ == "__main__":
    main()
