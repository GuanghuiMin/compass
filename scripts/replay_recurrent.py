"""Replay frozen episodes through the local-revision updater.

The same frozen inputs and the same integrity check as the baseline preflight; a different operator
and a different artifact root, so no run of one can ever be read as a run of the other.

`--verify-only` reads no environment variable, builds no client and creates no directory, so it
cannot spend anything.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from future_graph.adapter import from_environment                          # noqa: E402
from future_graph.episodes import MANIFEST_PATH, load                      # noqa: E402
from future_graph.revision_run import (                                    # noqa: E402
    ARTIFACT_ROOT, prepare_revision_run, replay_revision,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="directory holding the frozen episode files")
    parser.add_argument("--run-id", help="one path component under artifacts/recurrent/")
    parser.add_argument("--verify-only", action="store_true",
                        help="check every frozen input and stop, making no call")
    parser.add_argument("--episode", action="append", dest="episodes",
                        help="replay only this episode id, repeatable; default is all of them")
    args = parser.parse_args(argv)

    inputs = load(args.source_dir, MANIFEST_PATH)
    print(f"verified {len(inputs.episodes)} episodes, {inputs.boundary_count} boundaries")
    print(f"input manifest {inputs.input_manifest_sha256}")

    if args.verify_only:
        return 0

    if args.episodes:
        # Every frozen input was verified above, whether or not it is replayed: narrowing the run
        # must not narrow the integrity check.
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

    adapter = from_environment()
    prepared = prepare_revision_run(args.run_id, ARTIFACT_ROOT)
    print(f"claimed {prepared.run_dir} at {prepared.commit_sha}")
    manifest = replay_revision(inputs, adapter, prepared)
    print(f"status {manifest['status']}, completed {manifest['completed_boundaries']}, "
          f"accepted {manifest['accepted_boundaries']}")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
