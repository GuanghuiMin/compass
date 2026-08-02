"""Verify the frozen inputs, or replay them once through the regeneration operator.

Two things this deliberately does not offer: a way to point at another manifest, and a way to write
somewhere other than the artifact root. The loader takes paths so a test can use a fixture; the
command does not expose that door, because an experiment that can be aimed elsewhere is not the
experiment the specification describes.

`--verify-only` reads no environment variable, builds no client and creates no directory, so it
cannot spend anything.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from future_graph.adapter import from_environment            # noqa: E402
from future_graph.episodes import MANIFEST_PATH, load        # noqa: E402
from future_graph.run import ARTIFACT_ROOT, prepare_run, replay   # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="directory holding the frozen episode files")
    parser.add_argument("--run-id", help="one path component under artifacts/preflight/")
    parser.add_argument("--verify-only", action="store_true",
                        help="check every frozen input and stop, making no call")
    parser.add_argument("--episode", action="append", dest="episodes",
                        help="replay only this episode id, repeatable; default is all of them")
    args = parser.parse_args(argv)

    inputs = load(args.source_dir, MANIFEST_PATH)
    print(f"verified {len(inputs.episodes)} episodes, {inputs.boundary_count} boundaries")
    print(f"input manifest {inputs.input_manifest_sha256}")
    print(f"source_kind {inputs.source_kind}, "
          f"historical byte identity verified: {inputs.historical_byte_identity_verified}, "
          f"deterministic sampling: {inputs.sampling_is_deterministic}")

    if args.verify_only:
        return 0

    if args.episodes:
        # Every frozen input was verified above, whether or not it is replayed: narrowing the run
        # must not narrow the integrity check. The run manifest takes its episode order and its
        # boundary counts from what is left, so the artifact says which episodes this run covered
        # and never implies the others were attempted.
        known = {episode.id for episode in inputs.episodes}
        unknown = [name for name in args.episodes if name not in known]
        if unknown:
            parser.error(f"no such episode: {', '.join(unknown)}")
        inputs = replace(inputs, episodes=tuple(episode for episode in inputs.episodes
                                                if episode.id in set(args.episodes)))
        print(f"replaying {len(inputs.episodes)} of {len(known)} episodes, "
              f"{inputs.boundary_count} boundaries: "
              f"{', '.join(episode.id for episode in inputs.episodes)}")

    if not args.run_id:
        parser.error("--run-id is required unless --verify-only")

    # The adapter is settled first. A directory claimed before the environment was checked is an
    # empty run nobody can use under a run id nobody can reuse.
    adapter = from_environment()
    prepared = prepare_run(args.run_id, ARTIFACT_ROOT)
    print(f"claimed {prepared.run_dir} at {prepared.commit_sha}")
    manifest = replay(inputs, adapter, prepared)
    print(f"status {manifest['status']}, completed {manifest['completed_boundaries']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
