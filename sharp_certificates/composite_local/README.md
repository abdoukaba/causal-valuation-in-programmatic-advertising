# Composite dynamic auction equilibrium

Supplement for Abdou Kaba's *Causal Valuation and Equilibrium Uniqueness in Adaptive First-Price Auctions: A Mean-Field Analysis*. This study uses the archived iPinYou–Criteo calibration and rounded solver paths. Its primary game uses Shannon entropy over feasible actions and a unit horizon.

## Reproduce

Python 3.12.14, Bash and a CPU were used. Run in a copy of the archive. All proof inputs needed for offline execution are provided. After the dependencies are installed, the experiment makes no network requests.

    python -m pip install -r requirements.txt
    bash reproduce.sh grid
    bash reproduce.sh continuous
    bash reproduce.sh all

The grid mode regenerates the primary center and verifies both the broad and narrow primary cubes, then checks the center defect by a separate interval implementation. The continuous mode recomputes the surrogate policy coupling, both local gain bounds, the finest dual residual and the transfers. It takes the shipped discrete proof results as inputs. All mode also reruns the full grid/refinement battery, original-solver derivative comparison, iPinYou integer ledgers and radii, dual refinements and repeated precision runs. It recreates the figure. On the original host the refined 240-coordinate sweeps took roughly a minute each. Allow several minutes and several GB of memory for all mode. Run serially. Logs or timings can vary by host. The fine-grid proof is memory intensive because it retains the backward policy derivatives. For the original hardware, the programs were run under an 8 GiB memory limit.

    pdflatex -interaction=nonstopmode -halt-on-error Composite_Equilibrium_Note.tex
    pdflatex -interaction=nonstopmode -halt-on-error Composite_Equilibrium_Note.tex

## Result files

- results/cert_21_60_r9_p224.json establishes local uniqueness on the primary grid. Its exact rational center is in results/center_21_60.json. Radius 2^-9 is a coordinate-sup radius. The tighter TV bound around the same center is in cert_21_60_r28_p128.json. The 2^-8 test was inconclusive.
- results/cert_31_120_r10_p128.json is the refined-grid test in a neighborhood containing both sampled solver paths. Each FD root pertains to its own discretization.
- results/continuous_dual_160_160_p160.json records the exact continuous-response residual against the archived 320-cell midpoint flow. The first 160 is the number of output blocks, the second the belief subdivisions, and p160 the precision in bits. results/transfer.json contains the continuous-response bounds for the published FEM and PINN paths and linear extensions of the discrete roots.
- results/continuous_tube* record the attempted continuous uniqueness tests on narrower coordinate boxes. The reported gains exceeded one. The separate instantaneous gain in the earlier Schauder existence argument applies with continuation and population fixed.
- audit_ipinyou/results contains the recalculated iPinYou proof quantities plus the composite fixed-policy coupling used for the dual residual. The two recalculated iPinYou radius files match the archived references in data byte for byte. The .30 and .45 integer ledgers were also checked separately.
- independent_center.py checks the primary center defect independently in mpmath.iv. empirical_check.py checks all derivative columns by finite differences of the original solver. The 224-bit Arb runs are same-algorithm precision replays. The audit guide identifies proof obligations for third-party review.
- The figure is supplied in vector PDF. The numerical results and tables are supplied as JSON. The report wrapper compiles the drop-in LaTeX section.

## Provenance and scope

The calibration and archived paths were recovered from PINN_FEM_Replication.zip, SHA-256 b8b56c5bc62120a8edc5b28e17f020d5bffb4e8d5440074d62f745a936853f78. The original archive's checksum manifest passed. The small FEM and PINN path files are lossless subsets of their stored integer nodal flows and scale. Source scripts under prior, the comparison in continuum/proof, and the Bellman verifier in audit_ipinyou are from that earlier archive. The one-shot interval prototype motivated the local test. All dynamic grid derivative calculations, dual affine tests and transfers here were computed anew. No raw advertising records were required. The composite relies on a structural combination of distinct datasets and assumed monetary values. SHA256SUMS.txt checks the shipped package before it is run.

## Fresh replay during this study

The primary grid mode was rerun in a separate working tree with an empty results directory. The center and independent mpmath defect record matched byte for byte. A separate continuous replay started from the frozen calibration, source witnesses and verified grid files. Its finest 160-bit continuous dual result and all transferred residuals matched byte for byte. The logs are included. These are clean environment-directory reruns with the same installed dependencies.
