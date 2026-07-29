#!/usr/bin/env python3
"""Create or verify the pinned V7 native-5B container receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wm3d_v3.data.scale5b_contracts import canonical_sha256
from wm3d_v3.training.scale5b_environment import (
    create_environment_receipt,
    verify_environment_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    creating = args.output is not None
    verifying = args.receipt is not None or args.expected_sha256 is not None
    if creating == verifying:
        parser.error("choose exactly one of --output or --receipt/--expected-sha256")
    if verifying and (args.receipt is None or args.expected_sha256 is None):
        parser.error("verification requires --receipt and --expected-sha256")
    return args


def main() -> None:
    args = parse_args()
    if args.output is not None:
        receipt = create_environment_receipt(
            contract_path=args.contract,
            output_path=args.output,
        )
    else:
        receipt = verify_environment_receipt(
            args.receipt,
            expected_sha256=args.expected_sha256,
            contract_path=args.contract,
            check_current=True,
        )
    print(
        json.dumps(
            {
                "pass": True,
                "receipt_sha256": canonical_sha256(receipt),
                "environment_fingerprint_sha256": receipt["environment"][
                    "fingerprint_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
