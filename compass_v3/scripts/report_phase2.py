"""Phase 2 report: every POST method against the shared PRE control.

    python scripts/report_phase2.py --root artifacts/verify_dev --out reports/phase2_dev.md

Per method and horizon k: mean dU/dE/dR with bootstrap CI (Trace definitions).
Also a paired comparison of each COMPASS variant against the recorded OpenClaw
POST on the same boundaries (mean difference in dU, bootstrap CI), which is the
gate quantity: the two POST arms share the PRE control, so the difference is
POST_method - POST_openclaw per boundary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/verify_dev")
    ap.add_argument("--out", default="reports/phase2_dev.md")
    ap.add_argument("--baseline", default="openclaw")
    a = ap.parse_args()
    from compass.eval.burden import HORIZONS, bootstrap_mean, scan

    root = Path(a.root)
    pre_rows = read_jsonl(root / "pre" / "rollouts.jsonl")
    methods = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name != "pre"
                     and (d / "rollouts.jsonl").exists())
    lines = [f"# Phase 2 boundary verification — `{root}`", ""]
    pb_by_method = {}
    for m in methods:
        bnds = json.loads((root / m / "boundaries.json").read_text(encoding="utf-8"))
        ident = {b["replay_identity"]: b["boundary_id"] for b in bnds}
        pre = [dict(r, boundary_id=ident[r["replay_identity"]]) for r in pre_rows if r["replay_identity"] in ident]
        post = read_jsonl(root / m / "rollouts.jsonl")
        res = scan(pre + post)
        pb_by_method[m] = res["per_boundary"]
        tokens = [(r.get("extra") or {}).get("summary_tokens") for r in post if r.get("extra")]
        tokens = [t for t in tokens if t]
        levels = [(r.get("extra") or {}).get("level") for r in post if r.get("extra")]
        deg = sum(1 for r in post if (r.get("extra") or {}).get("degraded"))
        head = f"## {m}  (boundaries={len(res['per_boundary'])}, POST rollouts={len(post)}"
        if tokens:
            head += (f", mean summary tokens={sum(tokens)/len(tokens):.0f}, "
                     f"levels={ {l: levels.count(l) for l in sorted(set(levels), key=str)} }, degraded={deg}")
        lines += [head + ")", "", "| k | dU | dE | dR | PRE_U | POST_U |", "|---|---|---|---|---|---|"]
        for k, row in res["table"].items():
            def f(d):
                return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{'*' if d['sig'] else ''}"
            lines.append(f"| {k} | {f(row['dU'])} | {f(row['dE'])} | {f(row['dR'])} | {row['dU_pre']:.2f} | {row['dU_post']:.2f} |")
        lines.append("")
    if a.baseline in pb_by_method:
        base = pb_by_method[a.baseline]
        lines += [f"## Paired difference vs `{a.baseline}` (POST_method − POST_{a.baseline}, same boundaries; negative = less burden)", "",
                  "| method | n | " + " | ".join(f"ΔdU@k{k}" for k in HORIZONS) + " |",
                  "|---|---|" + "---|" * len(HORIZONS)]
        for m, pb in pb_by_method.items():
            if m == a.baseline:
                continue
            common = sorted(set(pb) & set(base))
            cells = []
            for k in HORIZONS:
                diffs = [pb[b][k]["dU"] - base[b][k]["dU"] for b in common]
                mean, lo, hi = bootstrap_mean(diffs)
                cells.append(f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if (lo > 0 or hi < 0) else ''}")
            lines.append(f"| {m} | {len(common)} | " + " | ".join(cells) + " |")
        lines.append("")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
