# Certified computed equilibrium and discretization bounds (Sections 6-7, Appendices C-D)

This folder reproduces Theorem S1 and Corollary S1 (Supplemental 1, Section S1) and Corollary 2 of the article: the bound
||q_hat - q*||_{w2} < 0.0706 (eta = 0.50) and < 0.449 (eta = 0.60) at the benchmark,
with w2(t) = 1 + 1.99 t - 0.68 t^2 and theta = 14.

Requirements: Python 3.10+, python-flint 0.9.0 (Arb), mpmath 1.3, numpy.

## Contents

| Path | What it is |
|---|---|
| `candidates/candidate_eta0.50_n7.json`, `candidate_eta0.60_n7.json` | The archived decimal candidates (bid flow on 60 cells; cubic Hermite in t with degree-7 Chebyshev coefficients in p). They are part of the mathematical input. |
| `arb/` | Arb (128-bit ball arithmetic) enclosures of all inputs in Table S1. `arbshim.py` maps the interval interface onto `flint.arb`. |
| `arb/logs`, `arb/outputs` | Logs and per-box output of the runs reported in the paper. |
| `mpmath_crosscheck/` | Independent implementation of the same enclosures in mpmath interval arithmetic, with its saved per-cell outputs in `outputs/`. |
| `weight/quadw.py` | Arb enclosure of the weighted contraction constant kappa^{w2}_theta (theta = 14, 16), using `code/cert_interval.py`. |
| `ode_crosscheck/` | Floating-point (DOP853) solution of the exact ten-state chain; a cross-check of r_node, not part of the proof. |
| `approximation/` | Arb constants and bounds of Section 7 (time modulus, Hölder constants, convergence of the time- and belief-projected schemes; Table S2). |
| `weight/quad_thresh.py` | Bisection showing that the w2 criterion holds at eta = 0.61 and fails at 0.62. |

## Reproduction

```bash
cd certified_equilibrium/arb
python arb_hjb_check.py ../candidates/candidate_eta0.50_n7.json   # HJB defect delta_H over 600 boxes
python arb_rest.py      ../candidates/candidate_eta0.50_n7.json   # L_p, L_t, r_node (ten-state chain, 6000 steps)
# repeat with candidate_eta0.60_n7.json
python aggregate_arb.py                                            # final bounds 0.07051 / 0.44868
cd ../weight && python quadw.py                                    # kappa^{w2}_14 = 0.80540 / 0.96930
```

`aggregate_arb.py` contains the Arb upper ends printed by the runs above (see `arb/logs`) and combines
them with decimal arithmetic rounded upward. `arb_hjb_check.py` writes its per-box output next to the candidate file.
The mpmath cross-check gives slightly smaller enclosures (delta_H <= 1.22e-5, L_p <= 0.1737): run
`cert_chunk.py`, `cert_lip_chunk.py`, `cert_time_lip.py`, `forward_interval.py` and then `aggregate_quadratic_v8.py`
(which reads the saved files in `mpmath_crosscheck/outputs`).

The candidates were produced by a floating-point spectral solver. The certificate does not depend on that solver,
so it is not included; the bound is a posteriori and uses only the archived decimal coefficients.

For Supplemental 1, Section S3:

```bash
python certified_equilibrium/approximation/approx_consts.py   # alpha*, H*, J*, C_p (128-bit Arb)
python certified_equilibrium/approximation/approx_bounds.py   # bounds of Table S2 (upward decimal rounding)
```
