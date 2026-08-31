"""AppWorld action signatures and the blocked-action contract.

``canon`` and ``sigs`` are reproduced verbatim from nokia-applied-research/Trace
(``trace_cc/core.py``) so refetch coordinates stay comparable with that cohort.
"""
from __future__ import annotations

import re

APIRE = re.compile(r"apis\.(\w+)\.(\w+)\(")
ERROR_PREFIX = "Execution failed. Traceback:"
NO_CODE = "No code available to execute."


def is_error(observation: str | None) -> bool:
    o = observation or ""
    return o.startswith(ERROR_PREFIX) or o.strip() == NO_CODE


def canon(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    if not s:
        return ""
    parts, buf, d = [], [], 0
    for c in s:
        if c in "([{":
            d += 1
        elif c in ")]}":
            d -= 1
        if c == "," and d == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    if buf:
        parts.append("".join(buf))

    def kw(p: str) -> bool:
        for i, ch in enumerate(p):
            if (ch == "=" and (i + 1 >= len(p) or p[i + 1] != "=")
                    and (i == 0 or p[i - 1] not in "=!<>")):
                return True
        return False

    return ",".join([p for p in parts if not kw(p)]
                    + sorted(p for p in parts if kw(p)))


def sigs(code: str) -> list[str]:
    if not code:
        return []
    out: list[str] = []
    i = 0
    while True:
        mo = APIRE.search(code, i)
        if not mo:
            break
        s0 = mo.end() - 1
        dep, j = 0, s0
        while j < len(code):
            if code[j] == "(":
                dep += 1
            elif code[j] == ")":
                dep -= 1
                if dep == 0:
                    break
            j += 1
        if j >= len(code):
            i = mo.end()
            continue
        out.append(f"{mo.group(1)}.{mo.group(2)}({canon(code[s0 + 1:j])})")
        i = j + 1
    return out


def primary_signatures(code: str) -> tuple[str | None, str | None]:
    calls = sigs(code or "")
    if not calls:
        return None, None
    exact = calls[0]
    return exact, exact.split("(", 1)[0]
