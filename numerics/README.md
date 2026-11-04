# numerics/: solver convergence, neural cross-check and sampled response ratios

Produces Section S5 ("The finite-difference solver and an independent neural solution"), Table S3, Fig. S1 and Table S6 of Supplemental 1.
Floating-point computations only; nothing here enters the certified results.

| script | output | manuscript |
|---|---|---|
| `fd_solver.py` | prints benchmark values | Sect. S5.1 (implicit variant of `code/verify_auction.py`) |
| `uniqueness.py` | `uniqueness.json` | Table S3 (key `convergence`); multi-start and Jacobians on 41x120 |
| `w2dist.py` | `w2dist.json` | Sect. S5.1, distance of the 21x60 solution to the 321x960 solution in the w2 norm |
| `pinn.py 6` | `pinn.json`, `pinn.pt` | Sect. S5.2 (about 33 min on 2 CPU cores) |
| `figures.py`, `fig_dga.py` | `compare.json`, `Fig3.pdf` (= `ESM_Fig1.pdf`) | Sect. S5.2, Fig. S1 (run inside this folder) |
| `global_lip21.py` | `global_lip21.json` | Table S6 |
| `global_lip.py` | `global_lip.json` | supplementary: same test on 41x120, max-coordinate norm |

`fd_ref.npz` is the 161x480 reference solution used by the figure scripts.
Requirements: numpy, scipy, matplotlib, torch (CPU is enough).
Check against the archived solver: on 21x60 at the benchmark the implicit solver's flow differs from `code/verify_auction.py` (T=1, lambda=2.25, eps=0.05, eta=0.45) by at most 4e-8.
