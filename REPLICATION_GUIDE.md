# Replication notes and result map

These notes cover manuscript revisions. The [archive overview](README.md) points to the current results. Run commands from the directories specified in each section.

## Results in the revised analysis

| Manuscript result | Reproduction materials |
|---|---|
| Section 9.3, original continuous Shannon point at 45% | conditional_composite/README.md, Arb verifier, two full ledgers and integer replay |
| Section 9.2, certified ranges through 19.5% for the composite | sharp_certificates/comparison/ and REPRODUCE_REVISION.md |
| Section 9.3, Corollary 4 and Table 16, parameter bands, anchored-KL and joint refit | range_certificate/README.md and run_suite.py |
| Additional point meshes, pooled allocation, alternative tail, regularization weights and attributed credit | range_certificate/results/, including inconclusive runs |
| Section 8.2, Table 13; Supplemental 1, S11.1, disclosure and valuation | code/counterfactuals.py and results/counterfactuals_*.json |
| Section 8.2, Table 14, alternative references at their own population | code/kl_ref_table.py and results/kl_ref_table.json |
| Uniform last-opportunity disclosure gain | range_certificate/disclosure_bound.py and results/disclosure_bound.json in that folder |
| Solver residuals and grid checks | sharp_certificates/ and Supplemental 1, S7 |

From range_certificate/, run python -S run_suite.py for an integer replay and python run_suite.py --regenerate to reconstruct the new enclosures. From results/, run python ../code/counterfactuals.py 41 160 for the finer-grid economic results. See range_certificate/README.md for the second grid, environment and limits. 

## Scope and provenance

Empirical evidence uses existing Criteo Uplift v2.1 user records and iPinYou auction logs. The separate mathematical certificate benchmark uses explicitly chosen primitives; those calculations are not empirical data. Calibrated equilibrium profit and causal-selection decompositions are conditional model calculations, not observed returns.

The raw Criteo and iPinYou datasets are not included. The supplied record-level resampling output is retained unchanged in results/ref_learning_real.json. This revision reconstructs its exact counts and rates and reproduces the calibrated model calculations from stored fitted primitives. Those earlier empirical checks did not refit the raw records. The continuous interval certificate added in this revision is documented above.

## Quick reproduction without raw data

From this package directory, run:

```bash
python code/audit_real_data.py --output-dir runs/audit --figure runs/audit/reference_selection_real_data.pdf --verify-model
python tests/test_kl_policy.py
```

The audit needs numpy, scipy and matplotlib. This command writes real_reference_audit.json and calibrated_model_recheck.json in runs/audit/. The figure is identical in content to the supplement figure. All 14 model comparisons (seven references, two grids, one independently rerun starting flow) reproduce the supplied results to less than 1.1e-12. The supplied four-start spreads remain separate evidence; the audit does not recheck four starts.

The Wilson intervals quantify Monte Carlo draw frequencies conditional on the fixed datasets and retained nuisance estimates. They are not intervals for performance in a new market.

## Manuscript to evidence map

| Material | Scripts and saved results |
|---|---|
| Terminal budget at the benchmark (Section 2.4) | `code/benchmark_budget.py`; `results/benchmark_budget.json` |
| Certificate benchmark, interval uniqueness and entropy trade-off | v40_figs.py, grid_certificate.py, cert_interval.py, envelope_gap_045.py; corresponding JSON files |
| Conditional uniqueness for anchored play | kl_ref_table.py, cert_calib.py; kl_ref_table.json |
| Fixed-flow KL deviation inequalities | tests/test_kl_policy.py; finite-action numerical checks |
| Finite-population bound | finite_n_v5.py; finite_n_v5.json (constants of the coupling proof, two-sided bounds of Table 8; finite_n_v4.py evaluates the earlier, looser bound) |
| Calibrated selection decomposition | calib_full.py; calib_full.json / criteo_offsets / selection_split |
| iPinYou threshold-based allocation proxy and descriptive price-response patterns | season1_day.py, calib_season1.py, ctr_bins.py, s3_summary.py; saved outputs |
| Criteo exposure estimates and fitted groups | load_criteo.py, criteo_aipw.py, criteo_regimes.py, criteo_diag.py; saved outputs |
| Robustness to the data inputs (Section 8.2) | sensitivity_inputs.py; sensitivity_inputs.json |
| Journal figures (Fig1-3.pdf, ESM_Fig1.pdf; vector, Arial-compatible font) | figs_dga.py (inputs: eps_table_arb.json, eps_tradeoff.json, envelope_gap_045.json, season1_day.json, calib_season1.json, ref_learning_real.json, and pay_won.npy from the raw 2013-03-16 logs) |
| Weighted uniqueness certificates, Theorem 1 and Tables 3-5 (Section 5) | cert_interval.py (sharp_kappa with weight w(t)=1+alpha t; alpha=0 is the scalar criterion), cert_weighted_all.py -> cert_weighted_all.json (all tables, minimized over theta and alpha), cert_weighted_fitted.py -> fitted_weighted.json (fitted iPinYou score scale), certify_weighted_rational.py (independent exact-rational check of the eta=0.60 certificate), ratio_eps004.py -> ratio_eps004.json (best-response ratios at eps=0.04, 0.03) |
| Latent rival-bid sensitivity under additional censoring assumptions (Section 8.2) | floor_censored.py (raw bid/imp/clk logs of 2013-03-16; censored Kaplan-Meier law, join check) -> floor_censored.json, floor_censored.npz; floor_censored_calib.py (refit zeta, Arb bounds, re-solved calibration) -> floor_censored_calib.json; floor_sensitivity.py gives the earlier worst-case bounds |
| Arb bounds across entropy weights (Table 4, Fig. 2 left) | cert_eps_table.py; eps_table_arb_fresh.json (the original eps_table_arb.json is kept) |
| Arb bounds at the iPinYou-fitted allocation (Table 10), without the raw log | cert_season1_only.py; calib_season1_cert_fresh.json |
| Arb bounds on the wider entropy grid (Section 5) | cert_eps_grid.py; eps_grid_arb.json |
| Calibrated profit table, both grids | calib_anchored_grid.py; calib_anchored_grid.json; independently reproduced by audit_real_data.py |
| Real-record resampling | ref_learning_real.py; original ref_learning_real.json |
| Counts including failures, conditional/unconditional rates and figure | audit_real_data.py; real_reference_audit.json |

## Full raw-data workflow

The original scripts use a working directory containing their inputs and write JSON into that working directory. To avoid overwriting packaged results, use a separate run directory and add the absolute code directory to PYTHONPATH. Copy the packaged results into that run directory when they are inputs to later computations. Store fresh outputs separately until compared with the retained originals.

1. Obtain the exact public Criteo v2.1 file. The recorded checksum is in results/criteo_sha256.txt. Run load_criteo.py with its CSV/gzip path to create criteo.npz.
2. Run criteo_regimes.py with criteo.npz. It creates criteo_regimes.json and criteo_tauW.npy. Run the AIPW and diagnostic scripts with the inputs indicated in their source.
3. Obtain iPinYou training1st logs for 20130316; expose the decompressed bid, impression, click and conversion files as bid.txt, imp.txt, clk.txt and conv.txt in the run directory. season1_day.py now explicitly emits ipy_s1.npz (won, pay, clk), the input required by ref_learning_real.py. Clicks for losses are not observed and are never treated as measured loss outcomes; the resampling script uses clicks only among wins.
4. Run ctr_bins.py and the calibration scripts. The score-scale calibration also uses pay_won.npy from season1_day.py. For the other empirical exercises, use training3rd logs for 20131023 and the season-one leaderboard file as documented in the corresponding scripts.
5. Run ref_learning_real.py with criteo.npz, criteo_tauW.npy, ipy_s1.npz and the required calibration JSONs in the run directory.

The broader code requires Python 3.10+, numpy, scipy, pandas, scikit-learn, matplotlib and python-flint. Interval calculations require python-flint/Arb. All interval certificates were re-run with python-flint 0.9.0 and reproduce every certified/not-certified verdict in the paper; see REPRODUCE_ARB.md for the environment, commands and logs.

## Optional acceleration (numba and JAX)

Two optional packages speed up the floating-point numerics. Install them with `pip install -r requirements-fast.txt`. Without them, every script runs the original numpy code.

- **numba** compiles three inner loops of the equilibrium solver in `fast_kernels.py`: the auction scores with belief interpolation, the Gibbs (softmax) step, and the mass redistribution after an auction epoch. `verify_auction.py` and `calib_model.py` call these kernels when numba is present. One evaluation of the response map is 6 to 8 times faster. The results agree with the numpy path to about 1e-16, and Picard iteration takes the same number of steps. Set `NO_NUMBA=1` to force the numpy path.
- **JAX** gives the exact Jacobian of the discretized response map by forward-mode autodiff (`jax_phi.py`). `index_test.py` and `index_norm.py` use it in place of one-sided finite differences with step 1e-6. One Jacobian takes about 0.15 s, against about 6 s for numpy finite differences (2 s with numba). It agrees with the finite-difference Jacobian to a relative error of about 1e-7, the truncation error of the finite differences. Set `JAC=fd` to force finite differences. `results/index_test.json` and `results/index_norm.json` now hold the autodiff values. The earlier finite-difference outputs are kept as `index_test_fd.json` and `index_norm_fd.json`, and they agree to every digit shown in Table 9.
- The interval certificates (`cert_interval.py`, `cert_general.py`, `cert_calib.py`) are unchanged. They use python-flint/Arb ball arithmetic, which numba and JAX cannot replace without losing the rigorous enclosure.

`python ../tests/test_accelerated.py`, run from `code/`, checks both accelerated paths against the numpy code; its output is saved in `results/test_accelerated.txt`. Rerunning `v40_figs.py`, `monotone_test.py`, `robustness.py` (including its Arb bounds), `calib_full.py` and `calib_anchored_grid.py` with numba reproduces the saved JSON to a relative difference below 6e-15.

## Primary dataset documentation

- Criteo release and erratum: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- Criteo authors' v2.1 repository: https://github.com/criteo-research/large-scale-ITE-UM-benchmark
- iPinYou competition data: https://contest.ipinyou.com/

## Fresh interval certification 
See REPRODUCE_ARB.md for versions, exact commands and logs.
All interval certificates were rerun with python-flint 0.9.0: cert_interval.py with 4000 cells, cert_eps_table.py, kl_ref_table.py, cert_season1_only.py, and the Arb parts of robustness.py and calib_full.py. Every certified/not-certified verdict in the paper is reproduced. Every printed bound equals the fresh upper bound or exceeds it by at most one unit in the last printed digit, because printed bounds are rounded up. A scan over theta in [8, 24], step 0.25, at eta = 0.487 certifies no value (smallest lower bound 1.00041 at theta = 16.25), which confirms the stated threshold.

## Certified computed equilibrium and convergence of discretizations (Sections 6-7)

See `certified_equilibrium/README.md`: the two archived decimal candidates, the Arb enclosures behind the certified-distance table, an independent mpmath cross-check, a floating-point ODE cross-check, the quadratic-weight scripts, and the constants of the convergence theorem (`approximation/`).


## Supplemental 1, Sections S5 and S6: solver convergence, neural cross-check, sampled ratios

`numerics/` contains an implicit variant of the finite-difference solver in `code/verify_auction.py` (they agree to 4e-8 on the 21x60 benchmark grid), the grid-refinement study (Table S3), the PINN with its particle simulation and trained weights (Fig. S1), and the sampled global ratios of Table S6. See `numerics/README.md`. These are floating-point computations.

## Section 9: sharp-allocation certificates

`sharp_certificates/` contains two things. The first is the comparison verifier with the η-ball sweeps behind Table 14. The second is the replication package for the composite grid certificate and continuous-game residuals in Section 9.3 and Supplemental 1, Section S7. See `sharp_certificates/README.md`. The comparison inequalities are derived in Appendix C of Supplemental 2 (`comparison_proof.pdf` is the earlier working note); the verifier evaluates them in outward-rounded Arb arithmetic.

## Calibrated-market checks added in v3 

Run from `results/` with `PYTHONPATH=../code`:

| Result (article, Section 8.2) | Script | Output |
|---|---|---|
| Profits of Shannon, anchored, always-h, always-2h and best-response policies against a common rival population (Shannon and anchored equilibrium flows) | `code/common_rival.py` | `results/common_rival.json` |
| Monetary-scale sensitivity (values x 0.5, 0.75, 1, 1.5) | `code/value_scale_coupling.py` | `results/value_scale_coupling.json` (its coupling block is a superseded grid search) |
| Exact bounds over all couplings of regime and rival bid: plug-in profits (vertices) and selection gap (closed-form optimization on every vertex-to-vertex segment, rational vertices, 60-digit evaluation, outward rounding) | `code/coupling_exact.py` | `results/coupling_exact.json` |
| Floor-censored law with the settings of Table 10 (weights 1+alpha t, same alpha/theta grids, 2000 cells) | `code/floor_cert_weighted.py` (with `code/cert_weighted_all_grid.py`) | `results/floor_cert_weighted.json` |

`results/floor_censored_calib.json` holds an earlier scalar evaluation (theta fixed at 16); its bounds (e.g. 1.36) are not comparable with Table 10.

## Terminal budget correction

Run `python code/benchmark_budget.py results/benchmark_budget.json` to reproduce the floating-point budget summary at the benchmark with 21 belief nodes, 60 time steps and eta=0.45. The mean terminal budget is 0.1481915089 from an initial budget of 0.20 (74.1% remaining), and the probability of exhaustion is 0.0441633416 (4.4%). These are grid estimates, not interval-certified quantities.

## Empirical interpretation corrections 

The baseline iPinYou curve estimates the strict CDF of C=max(R,F), the highest rival bid R combined with floor F, within the population entered at bid 300. It is not the raw rival-bid CDF. Winning thresholds are discretized into the exogenous model law q0; endogenous q remains a bid law. This is a structural transport assumption, including the imposed location of losing auctions in the tail. The baseline numerical inputs and certificates are unchanged. Price-bin click rates likewise use C among wins. Floor-censored results instead target R under independent censoring and the assumption that losses bound R at the logging bid; they are conditional sensitivity results. Historical JSON keys `cdf_naive` and `q0_naive` refer to the threshold baseline and are retained for compatibility.

For Criteo, common rescaling invariance is algebraic and does not establish robustness of the group-effect ratio to adjustment or assignment-design uncertainty. Pooled estimator sensitivities do not validate that ratio. The group multiplier-bootstrap errors condition on fitted nuisances and group labels; they do not refit the full learning pipeline. That correction restricted the claim to the evidence then available. The new review revision below adds a raw-data refit and group-specific sensitivity results.

The two figure generators now label the paying price as max(rival bid, floor). The supplied Fig3.pdf has the same vector curves and points with its axis label corrected; the archived data were not refitted.

## Revision: proofs, margins and group-specific Criteo results

See [REPRODUCE_REVISION.md](REPRODUCE_REVISION.md). Appendix C and the continuous-distance derivation are expanded. Precise sweep maxima, regenerated integer ledgers and Criteo group-specific unadjusted/IPW/OR/AIPW estimates, conditional ratio intervals and endpoint calibrations are now supplied. The fresh fit uses a new learned partition; the reference calibration and existing certificates retain their frozen inputs.
