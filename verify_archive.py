"""Verify archived hashes and replay the continuous-game certificate ledgers."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def verify_manifest(directory):
    entries = (directory / "SHA256SUMS.txt").read_text().splitlines()
    for entry in entries:
        digest, name = entry.split("  ", 1)
        with (directory / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"SHA-256 mismatch: {directory.name}/{name}")
    print(f"{directory.relative_to(ROOT)}: verified {len(entries)} checksums", flush=True)


def main():
    point = ROOT / "conditional_composite"
    extensions = ROOT / "range_certificate"
    for directory in (point, extensions, ROOT / "sharp_certificates/composite_local"):
        verify_manifest(directory)

    for mesh in ("240cells", "224bits"):
        stem = f"continuous_certificate_{mesh}"
        record = json.loads((point / f"{stem}.json").read_text())
        replay = subprocess.check_output(
            [sys.executable, "-S", "replay_weighted_integer.py", f"{stem}.ledger.json.gz"],
            cwd=point,
        )
        result = json.loads(replay)
        if (
            result["numerator"] != record["integer_aggregation_numerator"]
            or result["denominator"] != record["integer_aggregation_denominator"]
            or not result["strictly_below_one"]
        ):
            raise ValueError(f"Point certificate replay disagrees with {stem}.json")
        print(f'{mesh}: {result["numerator"]} / {result["denominator"]}', flush=True)

    subprocess.run([sys.executable, "-E", "-S", "run_suite.py"], cwd=extensions, check=True)


if __name__ == "__main__":
    main()
