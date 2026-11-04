"""Distance between the 21x60 FD equilibrium (the grid of the archived candidates) and the
converged FD solution, in the weighted norm ||q||_{w2}=sup_t ||q_t||_1/w2(t), w2=1+1.99t-0.68t^2."""
import json, numpy as np
from fd_solver import Params, FDSolver
out = {}
for eta in (0.5, 0.6):
    S0 = FDSolver(Params(eta=eta), 21, 60); q0, _ = S0.solve(tol=1e-12)
    S1 = FDSolver(Params(eta=eta), 321, 960); q1, _ = S1.solve(tol=1e-11)
    t = np.arange(60) / 60; qf = q1[::16]                # fine solution at the coarse nodes
    w2 = 1 + 1.99 * t - 0.68 * t**2
    d = np.abs(q0 - qf).sum(1)
    out[eta] = dict(w2_dist=float((d / w2).max()), l1_sup=float(d.max()))
    print(eta, out[eta], flush=True)
json.dump(out, open('w2dist.json', 'w'), indent=1)
