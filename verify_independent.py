#!/usr/bin/env python3
"""Repository entry point for the *independent* frozen-composite interval verifier.

--check validates artifact integrity only (stdlib).
--smoke also executes geometry/Bayesian tests (needs mpmath).
--full independently reconstructs every outer box, kernel and gain (needs mpmath).
--replay recomputes kernels/envelopes/gain from saved independent boxes, not end-to-end.

This deliberately does not import the Arb comparison implementation or its ledgers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
POINT = ROOT / "conditional_composite"
VERIFIER = POINT / "independent_verifier"
FROZEN = POINT / "v5" / "code" / "sharp_certificates" / "comparison" / "inputs"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_archive() -> None:
    """Only checks reproducibility material; this does NOT establish a bound."""
    manifest = VERIFIER / "SHA256SUMS.txt"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Verifier manifest is empty")
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = (VERIFIER / relative).resolve()
        if not path.is_relative_to(VERIFIER.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe verification file: {relative}")
        if digest(path) != expected:
            raise ValueError(f"Verifier checksum differs: {relative}")

    for filename in ("archived.json", "auction_fit.json"):
        a = VERIFIER / "data" / filename
        b = FROZEN / filename
        if digest(a) != digest(b):
            raise ValueError(f"Frozen primitive disagreement: {filename}")

    result = json.loads((VERIFIER / "data" / "verified_global_160.json").read_text())
    if (result.get("cells") != 160
            or result.get("verified_upper_1e12") != "975668389157"
            or result.get("strict_contraction") is not True):
        raise ValueError("Archived independent result does not match the supplied certificate")
    if result.get("primitive_archive_sha256") != digest(FROZEN / "archived.json"):
        raise ValueError("Archived result references different model primitives")
    if result.get("primitive_fit_sha256") != digest(FROZEN / "auction_fit.json"):
        raise ValueError("Archived result references different model fit")
    print(f"Independent archive: {len(lines)} checked files and matching frozen primitives", flush=True)
    print("Stored bound 0.975668389157 (reference only; run --full to recompute)", flush=True)


def run_smoke() -> None:
    subprocess.run([sys.executable, "-B", "test_independent_math.py"], cwd=VERIFIER, check=True)
    print("Independent smoke checks: PASS", flush=True)


def run_verification(*, recompute_outer: bool, output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-B", "independent_interval_replay.py",
            "--cells", "160", "--dps", "55", "--rounds", "8", "--out", str(output)]
    if recompute_outer:
        args += ["--steps", "8"]
    else:
        args += ["--boxfile", str(VERIFIER / "data" / "outer_boxes_independent.json")]
    subprocess.run(args, cwd=VERIFIER, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("cells") != 160 or result.get("strict_contraction") is not True:
        raise ValueError("Fresh independent replay did not verify the 160-cell contraction")
    boxes = result.get("outer_boxes_reconstructed", [])
    if len(boxes) != 9 or boxes[0] != [[0, 0, 0], [1000000, 1000000, 1000000]]:
        raise ValueError("Independent outer-box certificate missing or malformed")
    if int(result["verified_upper_1e12"]) >= 10**12:
        raise ValueError("Fresh independent bound fails to establish contraction")
    for name, provenance in (("archived.json", "primitive_archive_sha256"),
                             ("auction_fit.json", "primitive_fit_sha256")):
        if result.get(provenance) != digest(FROZEN / name):
            raise ValueError(f"Fresh replay used unexpected {name}")
    label = "full primitive-to-gain" if recompute_outer else "stored independent-box"
    print(f"Independent {label} replay: PASS", flush=True)
    print(f"Fresh outward upper bound: {int(result['verified_upper_1e12']) / 10**12:.12f}", flush=True)
    print(f"Fresh report: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Check packaged file hashes and primitive provenance only")
    mode.add_argument("--smoke", action="store_true", help="Check provenance and run independent geometry/Bayesian tests")
    mode.add_argument("--replay", action="store_true", help="Recompute 160-cell bound using saved independent outer boxes")
    mode.add_argument("--full", action="store_true", help="Independently recompute eight boxes and full 160-cell bound")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "independent_global_160.json",
                        help="Fresh output for --full/--replay (default: runs/independent_global_160.json)")
    args = parser.parse_args()
    check_archive()
    if args.smoke:
        run_smoke()
    elif args.full or args.replay:
        run_verification(recompute_outer=args.full, output=args.out)


if __name__ == "__main__":
    main()
