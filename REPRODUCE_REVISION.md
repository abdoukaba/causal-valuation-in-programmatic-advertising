Historical workflow for the calculations. For the revision and continuous certificates at the refit and selected anchor, use range_certificate/README.md.

# Reproduce the revised analysis 

The calculations compare empirical sensitivities around frozen reference inputs. The later joint-refit certificate is documented in [the continuous-game extensions](range_certificate/README.md).

## Environment

Python 3.12; numpy, scipy, pandas, scikit-learn **1.8.0**, joblib, numba and python-flint **0.9.0**. The new Criteo JSON records the exact Python/numpy versions and the fitted model hashes. Set `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for the Criteo fit. Classification uses scikit-learn's default early stopping with `max_iter=200`, not necessarily 200 completed iterations.

## Certificate margins and complete retained-ball sweep

Run from `sharp_certificates/comparison/`:

```bash
python replay_margins.py
python replay_margins.py --all
```

The first command recovers a rigorous maximum upper bound using the archived enclosing displays and fresh evaluations of every ball that could exceed the largest fresh bound. `results_margins/margin_summary.json` records the provenance of every bound, the coverage endpoints, numbers of balls, and margins. The second command replays **every retained ball in all six rows**, including all three extension sweeps, without reconstructing missing historical shell commands. Each retained interval's exact logged floating-point endpoints are converted to an Arb ball by the original midpoint and radius calculation, with radius multiplied by 1.000001. Each call uses two box restrictions, 40 cells, 224 bits and the row's entropy weight. The positive time weight is selected in floating point, then fixed to 12 decimal places before Arb evaluation.

TThe original fresh initial sweeps can also be run as follows:

```bash
N0=20 python sweep.py benchmark .05 0 1 40 benchmark_eps05.json
N0=20 python sweep.py benchmark .03 0 1 40 benchmark_eps03.json
N0=20 python sweep.py benchmark .02 0 1 40 benchmark_eps02.json
N0=20 python sweep.py benchmark .01 0 1 40 benchmark_eps01.json
N0=12 python sweep.py auction .05 0 .6 40 ipinyou_initial.json
N0=10 python sweep.py archived .05 0 .2 40 composite_initial.json
```

`sweep.py` internally fixes `ARB_BITS=224`, reads the entropy weight from its positional argument and bisects a failed interval at most four times. The supplied summaries retain the contiguous successful coverage. Printed compact Arb midpoint-radius strings can be much wider than the internal ball; `below_one` tests the internal ball, and the new `bound_upper` stores its upward decimal ceiling at 12 places.

## Regenerate detached integer ledgers

From `sharp_certificates/comparison/`:

```bash
ARB_BITS=224 EPS=.05 python proof/write_ledger.py .30 results_margins/ledger-auction-.30-40.json.gz
ARB_BITS=224 EPS=.05 python proof/write_ledger.py .45 results_margins/ledger-auction-.45-40.json.gz
python proof/replay_integer.py results_margins/ledger-auction-.30-40.json.gz
python proof/replay_integer.py results_margins/ledger-auction-.45-40.json.gz
```

The regenerated integer bounds are 4676772289/10^10 and 7900322858/10^10. The detached replay checks aggregation only; enclosure of the input constants still depends on Arb and Appendix C.

## Continuous iPinYou distances

From `sharp_certificates/composite_local/audit_ipinyou/`:

```bash
python code/validate.py results/witness_fem_ipinyou_0.45.json --bits 160 --parts 12
python code/validate.py results/witness_pinn_ipinyou_0.45.json --bits 160 --parts 12
python proof/tighten.py .45 ipinyou
```

The first two commands verify 320-cell residuals and membership in the invariant box. The last uses 224-bit arithmetic, the 40-cell comparison profile and 18 upper-bound updates. It reproduces 0.0517736002 and 0.0517471716 for the displayed paths, which the article rounds **upward** to 0.0517737 and 0.0517472. Supplemental 1 gives the induction and the exact rational interpolation transfer.

## Criteo group-specific analysis

Obtain the original v2.1 file from the authors' repository/dataset page or the Criteo organization mirror:

https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz

SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`.

From the root of this code archive:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/criteo_group_review.py /absolute/path/criteo-research-uplift-v2.1.csv.gz --out /absolute/path/group_run --draws 10000
```

The script verifies the checksum and row count, fits the original seed-3 two-fold grouping procedure, and computes unadjusted, IPW-only, outcome-regression-only and AIPW effects on the **same new groups**.  Original calibration inputs remain frozen in `results/criteo_regimes.json`. The new run is in `results/criteo_group_review/criteo_group_review.json`.

All fitted models and group joint-moment covariance summaries are included under `results/criteo_group_review/`. The script can resume when model and prediction checkpoints exist. To reuse the packaged fitted models, copy its eight `model_fold*.joblib` files into the new run directory before running. Raw data and row-level predictions are not redistributed in this package.

The 10,000-draw **Gaussian multiplier** bootstrap samples the joint moment mean through its equivalent multivariate normal distribution, then recomputes arm-normalized unadjusted effects, the other Wald effects and high/low ratios. It preserves numerator/denominator covariance. It holds the fitted nuisances and group assignments fixed, excludes full-pipeline and assignment-design uncertainty, and uses the same groups for clipping sensitivities at .01, .02 and .05. Signed regression-only estimates are retained, including a negative low-group effect. The narrow regression-only conditional interval is not a claim of precise causal identification.

The conditional intervals can also be replayed without raw data or fitting:

```bash
python code/replay_criteo_moments.py results/criteo_group_review
```

For calibration endpoint checks, run from `results/`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ../code/criteo_group_calibration.py criteo_group_review/criteo_group_review.json criteo_group_calibration.json
```

This recomputes each estimator's point ratio and both ratio-interval endpoints, preserving the sign of its group effects and retaining the reference exposure rates, prior and monetary normalization. Separate `joint_inputs` runs update both effects, both exposure rates and the new group share. Every Shannon and anchored equilibrium is computed from three starts on the 21x60 grid. The ratio endpoint cases are **not** certified extrema over a joint confidence region, nor do they carry a uniqueness certificate. All 16 input cases select bid h.

## Numerical conventions reconciled

The calibration click ratios 1:1.79242:2.55655 use all 1,395,945 impression records, repeated IDs included, with payments equal to 300 in the top bin. The floor comparison uses distinct joined wins and excludes payment 300; its upper-bin ratio is 2.53. The difference is documented in the article and does not silently alter the frozen payoff array. The printed prior 0.5 is rounded; the archived composite uses 0.4999294686139624.
