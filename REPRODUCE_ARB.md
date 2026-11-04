# Reproducing the interval (Arb) certificates

Last full rerun: 29-30 October 2026, on Linux with Python 3.13.18, python-flint 0.9.0 (Arb), numpy 2.5.3 and scipy 1.18.1.
The console output of every run is in `results/arb_logs/`.

Run the commands from `code/`. Each certificate takes seconds.

| Result in the paper | Command | Output | Log |
|---|---|---|---|
| Corollary 1, Table 3 (eta grid, 4000 cells), kappa at eta = 0.45, short horizon | `python cert_interval.py 4000` | `cert_interval.json` | `log_cert_interval.txt` |
| Threshold: no theta in [8, 24] (step 0.25) verifies at eta = 0.487 | `python theta_scan_eta0487.py` | (console) | `log_theta_scan_eta0487.txt` |
| Table 4 (entropy weights) | `python cert_eps_table.py 4000` | `eps_table_arb_fresh.json` | `log_cert_eps_table.txt` |
| Section 5, wider entropy grid | `python cert_eps_grid.py` | `eps_grid_arb.json` | `log_cert_eps_grid.txt` |
| Table 9 (anchored) | `python kl_ref_table.py` | `kl_ref_table.json` | `log_kl_ref_table.txt` |
| Table 10 (iPinYou-fitted allocation) | `python cert_season1_only.py` | `calib_season1_cert_fresh.json` | `log_cert_season1_only.txt` |
| Table 5 (robustness, Arb column) | `python robustness.py` | `robustness.json` | |
| Section 8.2 (calibrated market, Arb bounds) | `python calib_full.py` | `calib_full.json` | |

How to read the outputs: each value is a midpoint-radius ball [m +/- r], and "certified" means the whole ball lies below one. The paper prints upper bounds rounded up, so a printed value can exceed the fresh upper bound by one unit in the last digit. Every certified/not-certified verdict in the paper is reproduced by this rerun.

The scripts need only python-flint for the certificates (`pip install python-flint`). The floating-point numerics also use numpy and scipy, and optionally numba and JAX (see README).

## Time-weighted criterion (Theorem 1)

`cert_interval.sharp_kappa(..., alpha=a)` bounds the weighted gain with w(t)=1+a*t; `alpha='0'` reproduces every scalar value above.

    cd <run dir with criteo_regimes.json, calib_season1.json, ctr_bins.json>
    PYTHONPATH=<repo>/code python3 <repo>/code/cert_weighted_all.py      # all tables -> cert_weighted_all.json
    PYTHONPATH=<repo>/code python3 <repo>/code/cert_weighted_fixups.py   # ratios, censored law, grid-edge cases
    python3 -I <repo>/code/certify_weighted_rational.py --eta .60 --cells 4000 --output eta060.json  # independent check

Expected: eta=0.60, theta=14, alpha=1 gives an upper bound 0.996299... (< 1); the rational check gives 0.996398... (theta=16).
