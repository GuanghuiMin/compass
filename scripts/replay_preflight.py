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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from future_graph.adapter import from_environment            # noqa: E402
from future_graph.episodes import MANIFEST_PATH, load        # noqa: E402
from future_graph.run import ARTIFACT_ROOT, prepare_run, replay   # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="directory holding the frozen episode files")
    parser.add_argument("--run-id", help="one path component under artifacts/preflight/")
    parser.add_argument("--verify-only", action="store_true",
                        help="check every frozen input and stop, making no call")
    args = parser.parse_args()

    inputs = load(args.source_dir, MANIFEST_PATH)
    print(f"verified {len(inputs.episodes)} episodes, {inputs.boundary_count} boundaries")
    print(f"input manifest {inputs.input_manifest_sha256}")
    print(f"source_kind {inputs.source_kind}, "
          f"historical byte identity verified: {inputs.historical_byte_identity_verified}, "
          f"deterministic sampling: {inputs.sampling_is_deterministic}")

    if args.verify_only:
        return 0
    if not args.run_id:
        parser.error("--run-id is required unless --verify-only")

    run_dir = prepare_run(args.run_id, ARTIFACT_ROOT)
    print(f"claimed {run_dir}")
    manifest = replay(inputs, from_environment(), run_dir)
    print(f"status {manifest['status']}, completed {manifest['completed_boundaries']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
