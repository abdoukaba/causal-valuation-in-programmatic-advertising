"""Sampled global contraction ratios of Phi_FD (21x60 grid, the paper's solver grid), in the
L^inf_t(l^1) norm used for kappa_theta; plus multi-start. Configurations outside every certificate."""
import json, numpy as np
from fd_solver import Params, FDSolver, report
rng = np.random.default_rng(7)
IP = np.array([0.402, 0.255, 0.020, 0.323])
cases = [('benchmark, eta=0.5, eps=0.05', dict(eta=0.5)),
         ('benchmark, eta=1, eps=0.01', dict(eta=1.0, eps=0.01)),
         ('benchmark, eta=1, eps=0.005', dict(eta=1.0, eps=0.005)),
         ('iPinYou, zeta=0.011, eta=0.45, eps=0.05', dict(q0=IP, zeta=0.011)),
         ('iPinYou, zeta=0.011, eta=1, eps=0.05', dict(q0=IP, zeta=0.011, eta=1.0)),
         ('iPinYou, zeta=0.011, eta=1, eps=0.01', dict(q0=IP, zeta=0.011, eta=1.0, eps=0.01))]
def rand_flow(nt):
    k = rng.integers(3)
    if k == 0: return rng.dirichlet([0.2]*3, nt//6+1).repeat(6, 0)[:nt]
    if k == 1: return np.tile(np.eye(3)[rng.integers(3)], (nt, 1))
    return np.tile(rng.dirichlet([1]*3), (nt, 1))
n1 = lambda d: np.abs(d).sum(1).max()
out = []
for name, kw in cases:
    S = FDSolver(Params(**kw), 21, 60); L = []
    for i in range(200):
        a, b = rand_flow(60), rand_flow(60)
        if i % 2: b = a + 0.01*(b - a)
        den = n1(a - b)
        if den < 1e-12: continue
        L.append(n1(S.Phi(a) - S.Phi(b))/den)
    sols = [S.solve(q_init=q, damping=0.5, tol=1e-12)[0] for q in [np.tile(np.eye(3)[j], (60, 1)) for j in range(3)] + [rand_flow(60) for _ in range(9)]]
    spread = max(n1(s - sols[0]) for s in sols); r = report(S, sols[0])
    row = dict(case=name, pairs=len(L), lip_median=float(np.median(L)), lip_max=float(max(L)), starts=len(sols), spread=float(spread),
               J_eq=float(r['J_eq']), ratio=float(r['ratio']))
    out.append(row); print(row, flush=True)
json.dump(out, open('global_lip21.json', 'w'), indent=1)
