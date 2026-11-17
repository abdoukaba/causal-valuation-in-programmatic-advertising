# Independent numerical verifier — repository integration

## New implementation

The independent, `mpmath.iv`-based verifier is packaged in
[`conditional_composite/independent_verifier/`](conditional_composite/independent_verifier/).
Its code does not import the existing Arb interval pipeline or replay its
coefficient ledger. The verifier reads a separate, packaged copy of the exact
frozen model primitives, with SHA-256 checks against the original inputs.
It does use the stored **candidate time weights** as fixed rational witnesses;
the enclosures and weighted gain are computed afresh.

The new root command [`verify_independent.py`](verify_independent.py) supports:

- `--check`: hashes and input provenance only (standard library; **not** a proof replay).
- `--smoke`: above checks plus simplex and Bayesian consistency cases.
- `--replay`: fresh 160-cell gain calculation on *stored independent* outer boxes.
- `--full`: eight new invariant-box updates from the whole simplex, followed by
  new Bayesian, conditional, and gain enclosures (the full numerical pipeline).

The new [manual GitHub Actions job](.github/workflows/independent-composite.yml)
runs the archive check, smoke tests, and full check. Existing Arb code and
stored integer ledgers are not modified.

## Reproduction

From the repository root:

```bash
python -m pip install -r conditional_composite/independent_verifier/requirements.txt
python -S verify_independent.py --check
python verify_independent.py --smoke
python verify_independent.py --full
```

## Tests completed during this integration

| Check | Outcome |
| --- | --- |
| Frozen auction primitives in independent package versus original source | Byte-identical |
| SHA-256 independent archive/provenance check | Pass |
| Deterministic simplex LP cases | 160 passed |
| Diagnostic Bayesian posterior cases | 84 passed |
| Entire-simplex eight-step outer-set construction | Pass |
| Fresh 160-cell conditional and Bayesian kernel evaluation | Pass |
| Fresh independent weighted gain | **0.975668389157 < 1** |
| Agreement with deposited independent bound and 8-box sequence | Exact at archived reported precision |
| Original Arb 240-cell and 160-cell integer ledger replays | 0.961609277107 and 0.974202142743, both below 1 |
| Original `conditional_composite/SHA256SUMS.txt` | Pass after rehashing the intentionally edited README |
| Legacy `python -S verify_archive.py`, including all range cases | **PASS**, retaining two correctly inconclusive sensitivity cases |
| Independent `--check` and `--smoke` after range manifest reconciliation | **PASS** |

Fresh output saved in [`results/independent_global_160_reproduced.json`](results/independent_global_160_reproduced.json).
The verifier uses `mpmath==1.3.0` and 55 interval decimal digits.

## Scope

The computational verification independently implements the numerical comparison
bounds for the frozen Shannon model at competition share 0.45. The mathematical
proof of the comparison inequalities is still required. No formal verification
of the `mpmath` backend, external code audit, or parameter-uncertainty result is
claimed; this should not be generalized to the KL-anchored or unregularized game.

## Range-certificate checksum reconciliation

The three inherited mismatches in `range_certificate/SHA256SUMS.txt` have been
reconciled against the byte-identical original upload and current public GitHub
`main` files. Both retained experiment logs are explicitly inconclusive, as
expected by the archived gain JSON and integer replay. Their bytes are unchanged;
only the three stale manifest SHA-256 entries were updated. The old digests and
the evidence trail are retained in [ARCHIVE_HASH_RECONCILIATION.md](ARCHIVE_HASH_RECONCILIATION.md).

After this correction, the complete legacy archive verifier and the independent
archive/smoke checks were rerun (results documented in the reconciliation report).
A passing manifest check applies to the present deposited files; it does not imply
independent provenance of an earlier historical snapshot or full Arb regeneration.

The post-reconciliation full `--full` replay was rerun independently and returned
0.975668389157 < 1. Its output is included in
[`runs/independent_global_160_after_reconciliation.json`](runs/independent_global_160_after_reconciliation.json),
and the full log is retained under `verification_logs/independent_full.txt`.
