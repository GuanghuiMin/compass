"""Burden scoring (Trace's definitions, standard library only).

Per boundary b and horizon k:
    dE_b(k) = mean_E(POST,k) - mean_E(PRE,k)      blocked
    dR_b(k) = mean_R(POST,k) - mean_R(PRE,k)      refetch
    dU_b(k) = mean_U(POST,k) - mean_U(PRE,k)      union (a step counts once)
Reported as the mean over boundaries with a bootstrap 95% CI.
"""
from __future__ import annotations

import random
from collections import defaultdict

HORIZONS = [1, 2, 3, 4, 5]


def _count(roll: dict, key: str, k: int) -> int:
    return sum(1 for s in roll["steps"][:k] if s[key])


def per_boundary(rollouts: list[dict], arm_key: str = "arm") -> dict:
    """{boundary_id: {k: {"dE","dR","dU","nPRE","nPOST"}}}"""
    by = defaultdict(lambda: {"PRE": [], "POST": []})
    for r in rollouts:
        if r.get("error"):
            continue
        by[r["boundary_id"]][r[arm_key]].append(r)
    out = {}
    for bid, arms in by.items():
        if not arms["PRE"] or not arms["POST"]:
            continue
        out[bid] = {}
        for k in HORIZONS:
            row = {"nPRE": len(arms["PRE"]), "nPOST": len(arms["POST"])}
            for key, name in (("is_error", "dE"), ("refetch", "dR"), ("union", "dU")):
                pre = sum(_count(r, key, k) for r in arms["PRE"]) / len(arms["PRE"])
                post = sum(_count(r, key, k) for r in arms["POST"]) / len(arms["POST"])
                row[name] = post - pre
                row[name + "_pre"] = pre
                row[name + "_post"] = post
            out[bid][k] = row
    return out


def bootstrap_mean(values: list[float], n_boot: int = 2000, seed: int = 20260803) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    m = sum(values) / len(values)
    boots = []
    for _ in range(n_boot):
        s = [values[rng.randrange(len(values))] for _ in values]
        boots.append(sum(s) / len(s))
    boots.sort()
    return m, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def scan(rollouts: list[dict]) -> dict:
    pb = per_boundary(rollouts)
    table = {}
    for k in HORIZONS:
        row = {"n_boundaries": len(pb)}
        for name in ("dE", "dR", "dU"):
            vals = [pb[b][k][name] for b in pb]
            m, lo, hi = bootstrap_mean(vals)
            row[name] = {"mean": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                         "sig": bool(lo > 0 or hi < 0)}
            row[name + "_pre"] = round(sum(pb[b][k][name + "_pre"] for b in pb) / max(len(pb), 1), 4)
            row[name + "_post"] = round(sum(pb[b][k][name + "_post"] for b in pb) / max(len(pb), 1), 4)
        table[k] = row
    return {"per_boundary": pb, "table": table}


def format_table(table: dict) -> str:
    lines = [f"{'k':>2}  {'n':>4}  {'dU':>24}  {'dE':>24}  {'dR':>24}   PRE_U  POST_U"]
    for k, row in table.items():
        def f(d):
            return f"{d['mean']:+.3f}[{d['lo']:+.3f},{d['hi']:+.3f}]{'*' if d['sig'] else ' '}"
        lines.append(f"{k:>2}  {row['n_boundaries']:>4}  {f(row['dU']):>24}  {f(row['dE']):>24}  {f(row['dR']):>24}"
                     f"   {row['dU_pre']:.2f}   {row['dU_post']:.2f}")
    return "\n".join(lines)
