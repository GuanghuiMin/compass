"""How many refused graphs a single generic retry recovers.

This reads a finished run and asks one narrow question: if the model is told, without being told what
was wrong, to answer again in the required form, does the answer now pass the unchanged parser and
validator. It changes nothing at runtime -- no retry enters the operator, and the source run is opened
read-only.

The suffix is deliberately as weak as the one the natural-language planner uses: that planner appends a
fixed sentence and never names the fault, so naming ours would be a stronger intervention and a
different experiment.

Two cohorts, because they failed at different layers and must not be described alike. A parse rejection
never became a graph. A validation rejection parsed and then failed a constraint, so telling it that it
"could not be parsed" would be false, and a probe that lies to the model measures nothing.

Everything is checked before the first call and the whole thing refuses rather than skipping: a probe
that silently drops a boundary produces a rate over a denominator nobody chose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from future_graph.adapter import BASE_URL, MODEL, from_environment            # noqa: E402
from future_graph.artifacts import ModelCall, prompt_sha                       # noqa: E402
from future_graph.parser import parse                                          # noqa: E402
from future_graph.run import (                                                 # noqa: E402
    FROZEN_CONFIG, REQUIRED_OPENAI, check_openai_version, check_tree_clean, git, openai_version,
)
from future_graph.validation import validate                                   # noqa: E402

SOURCE_RUN = "frozen5_95f2993_r1"
SOURCE_COMMIT = "95f299336fed2dc282302a3eae99167d7ae53a05"
SOURCE_INPUT_MANIFEST = "29e9c03a8d36b48a00f12641c2e134661f3e8988e131f3992ae0a8a6aa94138d"
SOURCE_PROMPT_SHA = "e0a689d777adcd1a417731028a2a69e5f5cc63433b19d90e440dc8134f11f4f6"

PARSE_SUFFIX = (
    "\n\nYour previous output could not be parsed as the graph format described above.\n"
    "Return ONLY the graph, beginning with BEGIN_GRAPH and ending with END_GRAPH."
)
VALIDATION_SUFFIX = (
    "\n\nYour previous output was not accepted as a valid graph under the format and\n"
    "constraints described above. Return ONLY the graph, beginning with BEGIN_GRAPH\n"
    "and ending with END_GRAPH."
)

EXPECTED_PARSE = frozenset({
    ("042a9fc_3", 0), ("042a9fc_3", 1), ("042a9fc_3", 2),
    ("6b6ca61_2", 2), ("6b6ca61_2", 3), ("6b6ca61_2", 4),
    ("6f4b9a5_3", 0), ("6f4b9a5_3", 1), ("6f4b9a5_3", 5),
    ("83a7951_2", 1), ("83a7951_2", 9),
})
EXPECTED_VALIDATION = frozenset({("6b6ca61_2", 5), ("6f4b9a5_3", 6)})

PROBE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "repairability"


class ProbeError(RuntimeError):
    """A precondition of the probe that does not hold. Nothing is called."""


@dataclass(frozen=True)
class Rejection:
    episode: str
    index: int
    cohort: str                 # "parse" or "validation"
    empty_output: bool
    record: dict

    @property
    def key(self) -> tuple[str, int]:
        return (self.episode, self.index)

    @property
    def suffix(self) -> str:
        return PARSE_SUFFIX if self.cohort == "parse" else VALIDATION_SUFFIX


def discover(run_dir: Path) -> list[Rejection]:
    """Every refused boundary in a finished run, in episode and boundary order."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    found: list[Rejection] = []
    for episode in manifest["episode_order"]:
        for path in sorted((run_dir / f"episode_{episode}").glob("boundary_*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["accepted"]:
                continue
            found.append(Rejection(
                episode=episode, index=int(path.stem.split("_")[1]),
                cohort="parse" if record["parse_errors"] else "validation",
                empty_output=len(record["raw_output"]) == 0, record=record))
    return found


def verify_source(run_dir: Path, rejections: list[Rejection]) -> dict:
    """Refuse unless this is exactly the run the probe was written for."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for field, expected in (("status", "completed"), ("commit", SOURCE_COMMIT),
                            ("input_manifest_sha256", SOURCE_INPUT_MANIFEST),
                            ("prompt_sha", SOURCE_PROMPT_SHA), ("model", MODEL)):
        if manifest.get(field) != expected:
            raise ProbeError(f"source run {field} is {manifest.get(field)!r}, expected {expected!r}")
    if manifest.get("config") != FROZEN_CONFIG:
        raise ProbeError(f"source run config is {manifest.get('config')!r}")

    parse_keys = {r.key for r in rejections if r.cohort == "parse"}
    validation_keys = {r.key for r in rejections if r.cohort == "validation"}
    if parse_keys != set(EXPECTED_PARSE):
        raise ProbeError(f"parse rejections are {sorted(parse_keys)}, "
                         f"expected {sorted(EXPECTED_PARSE)}")
    if validation_keys != set(EXPECTED_VALIDATION):
        raise ProbeError(f"validation rejections are {sorted(validation_keys)}, "
                         f"expected {sorted(EXPECTED_VALIDATION)}")

    for r in rejections:
        call = r.record["model_call"]
        if prompt_sha(call["system"]) != SOURCE_PROMPT_SHA:
            raise ProbeError(f"{r.episode}#{r.index}: the stored system message does not hash to "
                             "the prompt this run declared")
        if {k: v for k, v in call["config"]} != FROZEN_CONFIG:
            raise ProbeError(f"{r.episode}#{r.index}: stored config is {call['config']!r}")
    return manifest


def retry_call(rejection: Rejection) -> ModelCall:
    """The original call, with one fixed sentence appended and nothing else changed."""
    call = rejection.record["model_call"]
    return ModelCall(system=call["system"],
                     user=call["user"] + rejection.suffix,
                     config=tuple((k, v) for k, v in call["config"]))


def judge(raw: str) -> dict:
    """The unchanged parser and validator decide, exactly as they did during the run."""
    outcome = parse(raw)
    if outcome.graph is None:
        return {"accepted_on_retry": False, "layer": "parse",
                "parse_errors": [[e.line, e.message] for e in outcome.errors],
                "violations": [], "snapshot": None}
    violations = validate(outcome.graph)
    return {"accepted_on_retry": not violations,
            "layer": "validation" if violations else None,
            "parse_errors": [],
            "violations": [[v.code, v.message, list(v.nodes)] for v in violations],
            "snapshot": outcome.graph.to_snapshot()}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict, overwrite: bool = False) -> None:
    if not overwrite and path.exists():
        raise ProbeError(f"{path} already exists")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def probe(run_dir: Path, out_dir: Path, model: Callable[[ModelCall], str],
          probe_commit: str) -> dict:
    """One retry per refused boundary, serial, recorded before the next is attempted."""
    rejections = discover(run_dir)
    verify_source(run_dir, rejections)
    out_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "source_run_dir": str(run_dir),
        "source_run_commit": SOURCE_COMMIT,
        "probe_commit": probe_commit,
        "source_input_manifest_hash": SOURCE_INPUT_MANIFEST,
        "source_prompt_hash": SOURCE_PROMPT_SHA,
        "parse_suffix": PARSE_SUFFIX,
        "parse_suffix_sha256": prompt_sha(PARSE_SUFFIX),
        "validation_suffix": VALIDATION_SUFFIX,
        "validation_suffix_sha256": prompt_sha(VALIDATION_SUFFIX),
        "model": MODEL,
        "base_url": BASE_URL,
        "config": dict(FROZEN_CONFIG),
        "openai_version": openai_version(),
        "planned": [f"{r.episode}#{r.index}" for r in rejections],
        "completed": [],
        "status": "running",
        "operational_failure": None,
        "started_at": _now(),
        "finished_at": None,
    }
    manifest_path = out_dir / "manifest.json"
    _write(manifest_path, manifest, overwrite=True)

    for r in rejections:
        call = retry_call(r)
        try:
            raw = model(call)
            if not isinstance(raw, str):
                raise ProbeError(f"a model returns text, got {type(raw).__name__}")
            verdict = judge(raw)
            _write(out_dir / f"boundary_{r.episode}_{r.index:03d}.json", {
                "episode": r.episode, "compaction_index": r.index, "cohort": r.cohort,
                "empty_original_output": r.empty_output,
                "original_parse_errors": r.record["parse_errors"],
                "original_violations": r.record["violations"],
                "original_raw_output": r.record["raw_output"],
                "suffix": r.suffix, "retry_raw_output": raw, **verdict,
            })
        except BaseException as err:
            manifest["status"] = "operational_failure"
            manifest["operational_failure"] = {
                "exception_type": type(err).__name__, "message": str(err)[:2000],
                "boundary": f"{r.episode}#{r.index}"}
            manifest["finished_at"] = _now()
            _write(manifest_path, manifest, overwrite=True)
            raise
        manifest["completed"].append(f"{r.episode}#{r.index}")
        _write(manifest_path, manifest, overwrite=True)

    manifest["status"] = "completed"
    manifest["finished_at"] = _now()
    _write(manifest_path, manifest, overwrite=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="one path component under the probe root")
    args = parser.parse_args(argv)

    if args.run_id != Path(args.run_id).name or args.run_id in ("", ".", ".."):
        parser.error("--run-id must be a single path component")

    check_openai_version()
    check_tree_clean()
    source = Path(__file__).resolve().parents[1] / "artifacts" / "preflight" / SOURCE_RUN
    out_dir = PROBE_ROOT / args.run_id
    if out_dir.exists():
        raise ProbeError(f"{out_dir} already exists")

    rejections = discover(source)
    verify_source(source, rejections)
    print(f"verified the source run and found {len(rejections)} refused boundaries "
          f"({sum(r.cohort == 'parse' for r in rejections)} parse, "
          f"{sum(r.cohort == 'validation' for r in rejections)} validation, "
          f"{sum(r.empty_output for r in rejections)} of them empty)")

    manifest = probe(source, out_dir, from_environment(), git("rev-parse", "HEAD"))
    print(f"status {manifest['status']}, completed {len(manifest['completed'])}"
          f"/{len(manifest['planned'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
