#!/usr/bin/env python3
"""Verify the portable offline-encoder bundle before array submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wm3d_v3.data.scale5b_assets import verify_asset_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    report = verify_asset_bundle(args.asset_root, deep=args.deep)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
