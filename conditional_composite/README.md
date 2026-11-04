# Reproduce Theorem 3

The exact game has Shannon regularization, a full initial budget, endogenous share 0.45 and fitted score scale 0.0110250693063193. The payoff and auction arrays are frozen in the inputs under v5/. Article Section 9.3 states the theorem. Appendix D of Supplemental 2 gives its inequalities and the dependencies in Appendix C.

Use Python and the packages in requirements.txt. Run from this directory.

    python conditional_arb.py restrict --cells 40 --rounds 6 --steps 8 --out outer_boxes_arb.json
    python conditional_arb.py verify --cells 240 --rounds 8 --boxfile outer_boxes_arb.json --out continuous_certificate_240cells.json
    python -S replay_weighted_integer.py continuous_certificate_240cells.ledger.json.gz
    python check_identities.py

The first command encloses every fixed point and records eight nested boxes. The second verifies the gain for the continuous response and writes a coefficient ledger. The third combines the upward-rounded rational bounds with standard-library integer arithmetic. It must return 961609277107 / 1000000000000. The identity check simplifies two derivative remainders to zero.

For the second partition, run

    python conditional_arb.py verify --cells 160 --rounds 8 --bits 224 --boxfile outer_boxes_arb.json --out continuous_certificate_224bits.json
    python -S replay_weighted_integer.py continuous_certificate_224bits.ledger.json.gz

The second integer upper bound is 974202142743 / 1000000000000. Either verified bound is sufficient for contraction on the invariant set. Time cells enclose integrals uniformly in continuous time. The proof uses no belief discretization or numerical candidate. The integer calculations audit the last aggregation stage. The analytical and elementary-function bounds are in the proof and Arb code.

The optional audit_numerical_envelopes.py and conditional_gain.py use floating point. They are diagnostics. File hashes for the deposited material are in SHA256SUMS.txt. Regeneration can change compressed file bytes because of gzip timestamps. Compare the decompressed ledger and the resulting integer bound when checking a fresh run.
