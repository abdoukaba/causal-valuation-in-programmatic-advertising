# Continuous-game extensions

The computations in this directory accompany Article Table 16, the point mesh sensitivity in Supplemental 2, and the disclosure and attribution comparisons in Supplemental 1. The frozen 240-cell and alternate 160-cell point proofs are in the sibling conditional_composite folder. The lower segment from 0 through 0.195 is documented separately under sharp_certificates/comparison, including its retained-ball coverage.

## Reproduction

Use Python 3.12.14. The arithmetic and numerical dependency versions are in requirements.txt. Run from this folder:

    python -m pip install -r requirements.txt
    python -S run_suite.py
    python run_suite.py --regenerate --jobs 2
    python disclosure_bound.py

The first check runs with standard-library Python only. It recomputes the SHA-256 of each compressed ledger and aggregates its upper coefficients with integer ceiling operations. Regeneration recalculates all outer boxes, policy envelopes, rational weights, kernel bounds and gain ledgers, including inconclusive cases. Optional positional case names select individual runs, for example:

    python run_suite.py --regenerate upper kl05 refit
    python -S replay_weighted_integer.py results/upper/gain.ledger.json.gz

The file results/reproduction.log records all tested cases and integer bounds. Each result subdirectory contains exact case metadata, the sequence of outer boxes, rational weights, an upper coefficient ledger, and arithmetic results. The normalized initial full-budget KL reference is (1,18,1)/20; the code renormalizes on affordable bids at each budget. All gains are for separate fixed games. Neither the band comparisons nor the two KL weights identify a shared equilibrium across different parameters.

## Inputs and variations

The exact input arrays are in v5/code/sharp_certificates/comparison/inputs. archived.json supplies the frozen composite. refit.json uses the jointly refitted primitives under aipw_joint_inputs in ../results/criteo_group_calibration.json. tail4h.json retains the values but places the top rival bid at 0.20. pooled.json retains the original regime values while allocation() replaces both win probabilities by their common prior average. attributed.json adds 0.02 to both value schedules when choosing bids. The regularized monetary comparison in ../code/counterfactuals.py subtracts that credit before measuring profit.

The parameter sweep checks three contiguous bands over all rival laws and all times. It does not evaluate the contraction only at endpoint games. Proof details, including the scaling of signed transfer bounds, are in Supplemental 2, D.7. At the lower Shannon weight and for the attributed game the reported bound exceeds one. Those runs are intentionally retained. The detached replay does not independently derive the analytic inequalities or enclose elementary functions.

## Economic experiments

From the sibling directory ../results run:

    python ../code/counterfactuals.py 21 60
    python ../code/counterfactuals.py 41 160

Use the versions recorded in environment.json for these solves. No raw user or auction records are needed for the model calculations. The JSON outputs give time-mean bid shares, incremental profit, monetary best responses and fixed-rival evaluations. The disclosure bound in this folder is a distinct outward-rounded check at a last opportunity with no continuation.
