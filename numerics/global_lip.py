"""Sampled global Lipschitz constant of the equilibrium map Phi (best response then forward law),
in the sup norm over (t,a), over random pairs of flows. sup < 1 over all samples is evidence
(not proof) of a global contraction, hence of uniqueness.  Also multi-start at sharp allocations."""
import json
import numpy as np
from fd_solver import Params, FDSolver, report

rng = np.random.default_rng(1)
cases = [
    ('benchmark', dict()),
    ('eta=1, eps=0.01', dict(eta=1.0, eps=0.01)),
    ('zeta=0.05, eta=1', dict(zeta=0.05, eta=1.0)),
    ('iPinYou q0, zeta=0.2, eta=0.45', dict(q0=np.array([0.402, 0.255, 0.020, 0.323]))),
    ('iPinYou q0, zeta=0.011 (fitted), eta=0.45', dict(q0=np.array([0.402, 0.255, 0.020, 0.323]), zeta=0.011)),
    ('iPinYou q0, zeta=0.011, eta=1, eps=0.01', dict(q0=np.array([0.402, 0.255, 0.020, 0.323]), zeta=0.011, eta=1.0, eps=0.01)),
]

def rand_flow(nt):
    kind = rng.integers(3)
    if kind == 0: return rng.dirichlet([0.2] * 3, nt // 6).repeat(6, 0)[:nt]
    if kind == 1: return np.tile(np.eye(3)[rng.integers(3)], (nt, 1))
    return np.tile(rng.dirichlet([1] * 3), (nt, 1))

out = []
for name, kw in cases:
    S = FDSolver(Params(**kw), 41, 120)
    L = []
    for _ in range(60):
        a, b = rand_flow(S.nt), rand_flow(S.nt)
        if rng.random() < 0.5: b = a + 0.01 * (b - a)        # nearby pairs too
        den = np.abs(a - b).max()
        if den < 1e-12: continue
        L.append(np.abs(S.Phi(a) - S.Phi(b)).max() / den)
    sols = [S.solve(q_init=rand_flow(S.nt), damping=0.5, tol=1e-10)[0] for _ in range(6)]
    spread = max(np.abs(s - sols[0]).max() for s in sols)
    r = report(S, sols[0])
    row = dict(case=name, lip_max=float(max(L)), lip_median=float(np.median(L)), spread=float(spread),
               J_eq=float(r['J_eq']), ratio=float(r['ratio']), max_dev=float(r['max_dev']))
    out.append(row); print(row, flush=True)
json.dump(out, open('global_lip.json', 'w'), indent=1)
