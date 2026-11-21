"""Pack full-prefill keys, C4 centroids and prefill scores for the live probe."""

import argparse
from pathlib import Path
import tempfile

import numpy as np

from gjallar_model_replay import (
    GROUPS,
    LAYERS,
    OLD_COUNT,
    SINKS,
    archive_blocks,
    centroid_mass_scores,
    load_capture,
    load_prefill_queries,
    quantized_centroids,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("kind", choices=("active", "archive"))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gjallar-archive-", dir=args.destination.parent) as scratch:
        temporary = Path(scratch) / args.destination.name
        with temporary.open("wb") as output:
            output.write(b"GJLIVE01")
            for layer in LAYERS:
                q, keys = load_capture(args.capture_dir, args.kind, layer)
                prefill = load_prefill_queries(args.capture_dir, args.kind, layer)
                _, history = centroid_mass_scores(q[:0], keys, prefill)
                history.astype("<f8").tofile(output)
                blocks = archive_blocks(keys)
                for group in range(GROUPS):
                    quantized_centroids(blocks[:, :, group]).astype("<f4").tofile(output)
                keys[SINKS:SINKS + OLD_COUNT].astype("<f2").tofile(output)
        temporary.replace(args.destination)


if __name__ == "__main__":
    main()
