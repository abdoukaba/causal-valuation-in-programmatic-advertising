# Sharp-allocation certificates (Section 9 of the article)

Two independent pieces of computation support Section 9, "A comparison certificate for sharp allocations".

## comparison/ — Table 14 (certified ranges of the endogenous share)

`proof/budget_gain.py` and `proof/enclose.py` compute the comparison gain of Section 9.1 with outward-rounded Arb arithmetic. `proof/comparison_proof.pdf` sketches the comparison inequalities. These files come from the earlier PINN/FEM replication archive, with three changes:

- The temperature is read from the `EPS` environment variable (default `.05`).
- The benchmark of Table 2 is available as the model `benchmark`.
- The share `eta` may be an Arb ball such as `[0.45 +/- 0.05]`, so one run covers an interval of shares. In the weight search, the ball's midpoint is used. In the interval run, the box restriction uses the lower and upper ends of each enclosure. Box faces of 0 and 1, which are trivially valid, are accepted.

`sweep.py` covers `[lo, hi]` with balls and bisects any ball that fails, up to four times:

    EPS=.05 N0=20 python sweep.py benchmark .05 0 1 40 out.json
    N0=12 python sweep.py auction .05 0 0.6 40 out.json     # iPinYou allocation
    N0=10 python sweep.py archived .05 0 0.2 40 out.json    # composite

The models are `benchmark` (Table 2), `auction` (iPinYou allocation with benchmark offsets) and `archived` (composite Criteo–iPinYou). All runs use 40 cells and 224 bits.

`results/uniqueness_sweep_summary.json` lists, for each row of Table 14:

- the certified balls covering `[0, eta_bar]`;
- the largest upper bound among them.

The `logs/` folder holds every ball tried, including the ones that failed. The iPinYou and composite sweeps were stopped once every ball beyond `eta_bar` had failed at the smallest width, so their summaries are built from the logs. The model `auction_raw` needs `inputs/raw_counts.json`. That file is derived from the raw iPinYou join and is not included, and no reported result uses it.

`proof/replay_integer.py` checks the integer-only aggregation for the two iPinYou ledgers using only the standard library:

    python proof/replay_integer.py proof/ledger-auction-.45-40.json.gz

**Status.** The Arb arithmetic is outward-rounded. The comparison inequalities that `budget_gain.py` evaluates are derived in Appendix C of Supplemental 2 (Lemmas C.1–C.4); `proof/comparison_proof.pdf` is the earlier working note.

## composite_local/ — Section 9.3 and Supplemental 1, Section S7 (composite calibration at eta = 0.45)

This is the replication package for the grid certificate and the continuous-game residuals, with its own `README.md`, `AUDIT.md` and `SHA256SUMS.txt`. Short checks:

    bash reproduce.sh grid
    bash reproduce.sh continuous

The grid certificate does not depend on the comparison inequalities. The transfer bounds and the iPinYou radii do depend on them.

## Extension sweeps and the existence constant (added 1 November 2026)

- `comparison/logs_ext/` and `comparison/results_ext/extension_summary.json`: second sweeps that start at the endpoints of the first sweeps, with balls of width about 0.002 bisected up to four times (same `sweep.py`, 224 bits, 40 cells). They extend the certified ranges of Table 14 to eta <= 0.838 (benchmark, eps = 0.01), eta <= 0.538 (iPinYou allocation) and eta <= 0.195 (composite). `comparison/extend_summary.py` recomputes the contiguous coverage from the logs. Point evaluations just beyond these endpoints exceed one (recorded in the summary).
- `comparison/proof/c_ex.py`: Arb enclosure of the existence constant of Proposition 7; `ARB_BITS=224 python c_ex.py archived 0.45` gives c_ex = 0.667785132864510 +/- 2.8e-16.
- `comparison/proof/replay_integer.py` replays the integer aggregation of the archived ledgers; the current `comparison/proof/write_ledger.py` regenerates both ledgers and reproduces their integer bounds. Complete margin, retained-ball replay, distance and empirical commands are in `../REPRODUCE_REVISION.md`.

## Precise margins and current reproduction

See `../REPRODUCE_REVISION.md` for the complete six-row retained-ball replay, the new integer-ledger writer, and the 18-step iPinYou distance derivation. `comparison/results_margins/margin_summary.json` contains explicit upper bounds and lower margins for every row.
