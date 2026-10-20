"""Prepare verified assets for the selected signal encoder."""

import argparse
import json
from pathlib import Path

from .vendor import signal


def download_encoder(directory, checkpoint=None):
    return signal.download_encoder(directory, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(f"--{signal.ASSET_KEY}", dest="directory", type=Path,
                        default=Path("third-party")/signal.ASSET_KEY)
    parser.add_argument("--checkpoint", type=Path, help="Optional local copy of the pinned upstream checkpoint")
    args = parser.parse_args()
    print(json.dumps(download_encoder(args.directory, args.checkpoint), indent=2))
