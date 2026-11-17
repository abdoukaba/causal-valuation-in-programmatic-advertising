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

Fresh output saved in [`results/independent_global_160_reproduced.json`](results/independent_global_160_reproduced.json).
The verifier uses `mpmath==1.3.0` and 55 interval decimal digits.

## Scope

The computational verification independently implements the numerical comparison
bounds for the frozen Shannon model at competition share 0.45. The mathematical
proof of the comparison inequalities is still required. No formal verification
of the `mpmath` backend, external code audit, or parameter-uncertainty result is
claimed; this should not be generalized to the KL-anchored or unregularized game.

## Pre-existing archive-integrity discrepancy (not modified)

The **original uploaded ZIP**, before these changes, already had three mismatches
in `range_certificate/SHA256SUMS.txt`:

- `range_certificate/README.md`
- `range_certificate/results/attributed.log`
- `range_certificate/results/eps03.log`

Their content and stored manifest hashes have been **left unchanged** to avoid
concealing an existing provenance inconsistency. Because of these inherited
mismatches, the legacy `python -S verify_archive.py` currently stops at the
range-certificate manifest check. This does not affect the separate new
`verify_independent.py --full` calculation, or the original point-ledger replays.
Reconcile these three files against their source-of-record before claiming that
the entire legacy archive-integrity suite passes.
