# Independent end-to-end interval replay (frozen composite uniqueness)

This implementation is integrated as `conditional_composite/independent_verifier/` in the
**Causal Valuation in Programmatic Advertising** replication repository. It is also
fully runnable from this folder without importing the existing Arb verifier.
For an integrated one-command reproduction from the repository root, run
`python verify_independent.py --full` after installing the requirements below.

**Verification boundary:** the interval computation is independently implemented,
but the comparison inequalities and envelope validity still rely on the proofs
in Supplemental 2. The standard `python -S verify_archive.py` check only replays
stored coefficient ledgers; it does not run this second implementation.


This archive independently reimplements the numerical enclosure pipeline of Supplemental 2, Appendix D for the frozen Shannon-regularized first-price auction model at competition share **eta = 0.45**. It reads **exact decimal primitives** from `data/archived.json` and `data/auction_fit.json`. It does **not** import the original Arb (`python-flint`), `conditional_arb.py`, `budget_gain.py`, or a computed coefficient ledger. Its only third-party dependency is `mpmath==1.3.0`, using the separate `mpmath.iv` directed-rounding interval arithmetic backend. Standard-library Python is used for the simplex enumeration, outward rational grid, and final upward rounding.

## Verified independent result

- Eight outer-set updates, starting from the entire three-action simplex, with 40 time cells and six finite conditional-envelope backward passes each: **PASS**.
- Final independent invariant-box enclosure for shares of bids (0, 0.05, 0.10): lower `[0.062958, 0.163962, 0.051761]`, upper `[0.685368, 0.845655, 0.564021]`.
- Reconstruct win/loss likelihoods, posterior maps, conditional-envelope recursions, Gibbs-policy bounds, signed forcing, budget domination, propagated-test kernels and forward/backward Duhamel integrals with 160 time cells and eight finite envelope passes: **PASS**.
- Independent verified weighted comparison, **upper-rounded to 12 decimals: 0.975668389157 < 1** (see `data/verified_global_160.json`). This is deliberately more conservative than the original Arb value, not a claim of numerical equality.
- The 160 rational weights in `data/weights_160.json` are merely fixed positive candidate weights initially obtained in the original numerical search. Their origin does not affect proof: **every coefficient and the gain for those weights are independently enclosed from the primitive inputs**. No original coefficient ledger or outputs are used in the proof calculation.
- Deterministic arithmetic/geometry smoke tests: 160 simplex-LP cases plus 84 posterior-point checks. The posterior checks alone do not certify a continuum; the continuum validity follows from the analytic convex-hull and positive fractional-linear Bayes formulas.

## Reproduce from a fresh Python environment

```bash
python -m pip install -r requirements.txt
python test_independent_math.py
python run_outer.py
python independent_interval_replay.py \
  --cells 160 --dps 55 --rounds 8 \
  --boxfile data/outer_boxes_independent.json \
  --out data/reproduced_global_160.json
```

Alternatively, a **single** command recomputes the outer boxes as well as the final gain (rather than loading the independently generated `outer_boxes_independent.json`):

```bash
python independent_interval_replay.py --steps 8 --cells 160 --dps 55 \
  --rounds 8 --out data/reproduced_one_command.json
```

The program returns a nonzero exit status if the upper-rounded contraction bound is not strictly below one, if outer boxes are incomplete, or if the primitive probabilities fail the positivity checks. `run_outer.py` writes the eight-box computation to `data/outer_boxes_independent.json`; re-running it replaces that file.

## What this verification does (and does not) establish

**Established computationally**, conditional on the mathematical comparison inequalities proved in Supplemental 2: a separately coded primitive-to-output interval calculation supports global uniqueness for the frozen stylized game. The time partition bounds continuous-time quantities; this is **not** an equilibrium computation on a belief grid. Strictly positive Bayesian likelihoods and their complements and the posterior derivative formulas hold for every belief in `[0,1]`, not just the diagnostic sample prior points.

**Not established**: an independently formalized derivation of every analytic envelope lemma; a verified compilation of `mpmath.libmpi`; third-party source-code audit; parameter uncertainty or a claim about actual advertisers' equilibria; analogous uniqueness for KL-anchored or unregularized bidding. This is a **second implementation-level audit**, not an independent mathematical theorem or external peer review.

The exact decimal calibration and assumptions are those stated in Supplemental 2. Numeric results in `data/verified_global_160.json` come from the independent run at 55 decimal digits. A different `mpmath` version or precision may produce slightly different conservative ceilings. Any value below one is independently sufficient.

## Repository-level checks

```bash
# From repository root (standard library only; verifies included source/output hashes)
python -S verify_independent.py --check

# Requires mpmath==1.3.0; runs geometric and Bayesian smoke tests
python verify_independent.py --smoke

# Requires mpmath==1.3.0; rebuilds the 8 boxes and the 160-cell contraction
# directly from bundled primitives, without loading the archived outer boxes
python verify_independent.py --full
```

The `--full` check writes a fresh result to `runs/independent_global_160.json` and exits
nonzero if its verified upper bound is not below 1. Use `--replay` to re-evaluate
from the *already independently constructed* boxes without re-running the
eight box updates; this shorter path is **not** the full end-to-end replay.
The `SHA256SUMS.txt` file lists the verifier's packaged inputs, source and prior
results. Its hashes establish archive integrity, not mathematical validity.
