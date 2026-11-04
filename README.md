# Causal Valuation in Programmatic Advertising: A Mean-Field Analysis

Replication code for the manuscript. The model studies how advertisers bid when the value of winning an impression depends on what the user would have done after a loss. This archive contains the Criteo–iPinYou fitted inputs, equilibrium solvers, verification code and reported numerical results.

## Verify the continuous equilibrium results

From the extracted folder containing this README, run the following with Python 3.12. This check uses the standard library and bundled files.

```bash
python -S verify_archive.py
```

The command checks the three distributed SHA-256 manifests, replays both Section 9.3 certificates for the frozen Shannon game at 45% endogenous competition, then replays all 13 cases in the extension suite. The two point bounds are **0.961609277107** (240 cells) and **0.974202142743** (160 cells, 224-bit arithmetic). Both are below one. In the extension suite, 11 bounds are below one. The lower Shannon weight of 0.03 and the attributed-value specification return inconclusive bounds as reported in the paper.

The replay checks the integer aggregation of the stored enclosures. To recompute the enclosures themselves, use the pinned Python dependencies and the Arb commands in the [point-certificate guide](conditional_composite/README.md) and [extension guide](range_certificate/README.md). Run regenerations in a separate extracted copy because some scripts overwrite the supplied results.

## Find a result

| Question | Code and evidence |
| --- | --- |
| Does the continuous composite have a unique equilibrium at 45% endogenous competition? | [Section 9.3 point verifier and ledgers](conditional_composite/README.md) |
| Which competition levels are covered? | [Comparison sweep through 19.5%](sharp_certificates/README.md) and [three contiguous bands from 19.5% through 45%](range_certificate/README.md) |
| What changes under anchored KL, a joint refit, or a different valuation rule? | [Parameter choices, ledger replays and counterfactuals](range_certificate/README.md) |
| How do selection into wins, disclosure and reference policies affect value or profit? | [Selection decomposition](results/selection_split.json), [disclosure and policy experiments](code/counterfactuals.py), and [calibrated reference-policy results](results/calib_anchored_grid.json) |
| How were the inputs fitted and the numerical approximations checked? | [Replication notes and raw-data workflow](REPLICATION_GUIDE.md), [group-specific estimates](REPRODUCE_REVISION.md), [certified equilibrium distances](certified_equilibrium/README.md), and [numerical solver comparisons](numerics/README.md) |

## Run the calibrated model checks

Create and activate a Python 3.12 virtual environment. For the finite-action policy tests and the model recheck below, install NumPy and SciPy. For broader replication, use the root requirements and the versions documented for the calculation you want to reproduce.

```bash
python -m pip install numpy scipy
python tests/test_kl_policy.py
python code/audit_real_data.py --output-dir runs/audit --verify-model
```

The second command checks the weighted Gibbs identity and policy bounds on 900 randomized cases. The third reconstructs counts from archived resampling summaries and compares 14 calibrated policy results across two grids to the stored outputs. Optional Numba and JAX acceleration is documented in the [replication notes](REPLICATION_GUIDE.md#optional-acceleration-numba-and-jax). The neural solver has its own [PyTorch instructions](numerics/README.md).

The raw Criteo user data and iPinYou auction logs are distributed separately by their providers. They are needed to refit from records. The certificates and model checks above use the fitted inputs shipped here. The [raw-data workflow](REPLICATION_GUIDE.md#full-raw-data-workflow) identifies the dataset versions, dates and Criteo checksum. One frozen input retains the historical source of the fitted parameters for provenance.

## Citation and license

Software citation metadata is in [CITATION.cff](CITATION.cff). The code is distributed under the [MIT License](LICENSE).
