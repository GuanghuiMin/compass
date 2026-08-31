"""Structured projection of observations: keep whole fields, not a character prefix.

A checkpoint line such as ``search_songs(...) -> [ { "song_id": 55, "title": "Tangled Lies", "album_id": 11, "dur...``
loses exactly the field the next step needs (release_date) while spending its budget on
fields nobody asked for.  ``project`` parses JSON-like observations and renders them field by
field within a width: identifier fields and fields named by the goal/plan first, then short
scalars, then everything else, with an explicit ``(+k more)`` marker instead of a silent cut.
Non-JSON text falls back to a plain excerpt.  Deterministic; no model call.
"""
from __future__ import annotations

import ast
import json
import re

ID_RE = re.compile(r"(^|_)ids?$")
NAME_KEYS = ("name", "title", "first_name", "email", "username", "account_name", "query")


def excerpt(text: str | None, n: int) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t if len(t) <= n else t[:n] + "..."


def parse(text: str | None):
    t = (text or "").strip()
    if not t or t[0] not in "[{":
        return None
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        try:
            return ast.literal_eval(t)
        except Exception:  # noqa: BLE001
            return None


def _scalar(v) -> bool:
    return v is None or isinstance(v, (int, float, bool, str))


def _fmt(v, w: int = 60) -> str:
    if isinstance(v, str):
        v = v.replace("\n", " ")
        return repr(v if len(v) <= w else v[:w] + "…")
    if isinstance(v, dict):
        inner = ", ".join(f"{k}={_fmt(x, 24)}" for k, x in list(v.items())[:4])
        return "{" + inner + (", …" if len(v) > 4 else "") + "}"
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(_scalar(x) for x in v):
            s = "[" + ", ".join(_fmt(x, 24) for x in v[:8]) + (", …" if len(v) > 8 else "") + "]"
            return s if len(s) <= w + 20 else f"<list n={len(v)}>"
        if all(isinstance(x, dict) for x in v):
            key = next((k for k in NAME_KEYS if k in v[0]), None)
            if key:
                return "[" + ", ".join(_fmt(x.get(key), 24) for x in v[:6]) + (", …" if len(v) > 6 else "") + "]"
        return f"<list n={len(v)}>"
    return str(v)


def _priority(key: str, val, focus: set[str]) -> tuple:
    toks = set(re.split(r"[_\W]+", key.lower())) - {""}
    is_focus = bool(toks & focus)
    is_id = bool(ID_RE.search(key))
    short = _scalar(val) and (not isinstance(val, str) or len(val) <= 40)
    long_text = isinstance(val, str) and len(val) > 200
    return (not is_focus, not is_id, not short, long_text)


def _dict(d: dict, width: int, focus: set[str]) -> str:
    keys = sorted(d, key=lambda k: _priority(k, d[k], focus))
    parts: list[str] = []
    used = 2
    for i, k in enumerate(keys):
        # scalars are short anyway; text fields (note content, descriptions) get whatever width is left,
        # in priority order, instead of a fixed cut
        w = 60
        if isinstance(d[k], str) and len(d[k]) > 60:
            w = max(60, width - used - len(k) - 6)
        p = f"{k}={_fmt(d[k], w)}"
        if used + len(p) + 2 > width and parts:
            parts.append(f"(+{len(keys) - i} more fields)")
            break
        parts.append(p)
        used += len(p) + 2
    return "{" + ", ".join(parts) + "}"


def _table(rows: list[dict], width: int, focus: set[str]) -> str:
    keys: list[str] = []
    for r in rows[:5]:
        for k in r:
            if k not in keys:
                keys.append(k)
    keys.sort(key=lambda k: _priority(k, rows[0].get(k), focus))
    # drop long free-text columns unless the goal names them
    keep = [k for k in keys if not (isinstance(rows[0].get(k), str) and len(rows[0][k]) > 120
                                    and not (set(re.split(r"[_\W]+", k.lower())) & focus))]
    head = f"{len(rows)} items: ["
    out, used = [], len(head) + 2
    for i, r in enumerate(rows):
        cells = []
        for k in keep:
            if k in r:
                cells.append(f"{k}={_fmt(r[k], 40)}")
        line = "{" + ", ".join(cells) + "}"
        if used + len(line) + 2 > width and out:
            out.append(f"… (+{len(rows) - i} more)")
            break
        out.append(line)
        used += len(line) + 2
    return head + "; ".join(out) + "]"


def project(text: str | None, width: int, focus: set[str] | None = None) -> str:
    """Render ``text`` (an observation) within about ``width`` characters, field-wise if it is
    JSON-like, otherwise as a plain excerpt."""
    focus = focus or set()
    obj = parse(text)
    if obj is None:
        return excerpt(text, min(width, 600))
    if isinstance(obj, dict):
        return _dict(obj, width, focus)
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(isinstance(x, dict) for x in obj):
            return _table(obj, width, focus)
        if all(_scalar(x) for x in obj):
            s = "[" + ", ".join(_fmt(x, 40) for x in obj) + "]"
            return s if len(s) <= width else s[:width] + f"… ({len(obj)} items)"
    return excerpt(text, width)


STOP = {"the", "and", "for", "with", "that", "this", "from", "into", "have", "them", "then", "also",
        "each", "want", "need", "please", "make", "sure", "using", "their", "which", "some", "only"}


def focus_tokens(*texts: str) -> set[str]:
    """Lower-cased content words (>=4 chars) of the goal / open plan: fields whose names share a
    token with them are rendered first."""
    toks: set[str] = set()
    for t in texts:
        for w in re.findall(r"[A-Za-z]{4,}", t or ""):
            w = w.lower()
            if w not in STOP:
                toks.add(w)
                if w.endswith("s"):
                    toks.add(w[:-1])
    return toks


def extract_fields(text: str | None, fields: list[str], max_items: int = 12) -> list[tuple[str, str]]:
    """Needs-conditioned extraction from a JSON-like observation: every (path, value) whose key
    matches one of ``fields`` (case-insensitive substring both ways). Values are rendered short."""
    obj = parse(text)
    if obj is None or not fields:
        return []
    want = [f.lower() for f in fields if f]
    out: list[tuple[str, str]] = []

    def walk(node, path):
        if len(out) >= max_items:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                kl = str(k).lower()
                if any(w in kl or kl in w for w in want) and _scalar(v):
                    out.append((f"{path}.{k}".lstrip("."), _fmt(v, 60)))
                    if len(out) >= max_items:
                        return
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for idx, v in enumerate(node[:20]):
                walk(v, f"{path}[{idx}]")

    walk(obj, "")
    return out
