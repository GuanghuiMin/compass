"""End-to-end report over episode directories.

    python scripts/report_episodes.py --root artifacts/dev41 --out reports/episodes_dev41.md

For each method dir under --root: Acc (success rate), mean score, mean steps,
max-steps rate, refetch/blocked per episode, peak prompt tokens, compaction
count, compressor tokens. If a second run dir (--root2) is given with the same
methods, Pass^2 and Pass@2 are reported over tasks present in both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(d: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[r["task_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--root2", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    root2 = Path(a.root2) if a.root2 else None
    lines = [f"# End-to-end results — `{root}`" + (f" + `{root2}`" if root2 else ""), "",
             "| method | n | Acc | mean score | Pass^2 | Pass@2 | steps | max-steps | refetch/ep | blocked/ep | peak prompt tok | compactions/ep | compressor tok/ep |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        eps = load(d)
        if not eps:
            continue
        n = len(eps)
        acc = sum(e["success"] for e in eps.values()) / n
        eps2 = load(root2 / d.name) if root2 and (root2 / d.name).exists() else {}
        both = [t for t in eps if t in eps2]
        p2 = sum(eps[t]["success"] and eps2[t]["success"] for t in both) / len(both) if both else None
        pa2 = sum(eps[t]["success"] or eps2[t]["success"] for t in both) / len(both) if both else None
        comp = [e["compressor_usage"]["prompt_tokens"] + e["compressor_usage"]["completion_tokens"]
                for e in eps.values() if e.get("compressor_usage")]
        lines.append("| {m} | {n} | {acc:.1%} | {sc:.2f} | {p2} | {pa2} | {st:.1f} | {mx:.0%} | {rf:.1f} | {bl:.1f} | {pk:.0f} | {nb:.1f} | {ct:.0f} |".format(
            m=d.name, n=n, acc=acc, sc=sum(e["score"] for e in eps.values()) / n,
            p2=f"{p2:.1%}" if p2 is not None else "-", pa2=f"{pa2:.1%}" if pa2 is not None else "-",
            st=sum(e["n_steps"] for e in eps.values()) / n,
            mx=sum(e["termination_reason"] == "max_steps" for e in eps.values()) / n,
            rf=sum(e["n_refetch"] for e in eps.values()) / n, bl=sum(e["n_blocked"] for e in eps.values()) / n,
            pk=sum(e["peak_prompt_tokens"] for e in eps.values()) / n,
            nb=sum(len(e["boundaries"]) for e in eps.values()) / n,
            ct=(sum(comp) / len(comp)) if comp else 0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
