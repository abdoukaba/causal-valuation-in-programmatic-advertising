"""Numerical evidence on existence/uniqueness/stability of the equilibrium."""
import json, time
import numpy as np
from fd_solver import Params, FDSolver, report

out = {}
rng = np.random.default_rng(0)

# 1. grid convergence at the benchmark
conv = []
for n_p, nt in [(21, 60), (41, 120), (81, 240), (161, 480), (321, 960)]:
    S = FDSolver(Params(), n_p, nt); q, h = S.solve(tol=1e-10)
    r = report(S, q); r.update(n_p=n_p, nt=nt, iters=len(h))
    r['q_coarse'] = q[np.arange(60) * nt // 60].tolist()
    conv.append(r); print('conv', n_p, nt, {k: r[k] for k in ('J_eq', 'J_br', 'ratio', 'max_dev')})
qf = np.array(conv[-1]['q_coarse'])
for r in conv: r['dq_vs_finest'] = float(np.abs(np.array(r['q_coarse']) - qf).max())
out['convergence'] = conv

# 2. multi-start + Jacobian spectral radius over (eta, eps)
def starts(nt):
    L = [np.tile([1, 0, 0], (nt, 1)).astype(float), np.tile([0, 1, 0], (nt, 1)).astype(float),
         np.tile([0, 0, 1], (nt, 1)).astype(float), np.full((nt, 3), 1 / 3)]
    for _ in range(8):
        x = rng.dirichlet([0.3] * 3, size=nt // 10 + 1).repeat(10, 0)[:nt]; L.append(x)
    return L

def jacobian_radius(S, q):
    nt = S.nt; base = S.Phi(q); n = 2 * nt; J = np.zeros((n, n)); hstep = 1e-6
    for j in range(n):
        k, a = divmod(j, 2)
        dq = np.zeros_like(q); dq[k, a] = hstep; dq[k, 2] = -hstep    # stay on simplex
        J[:, j] = ((S.Phi(q + dq) - base)[:, :2] / hstep).ravel()
    ev = np.linalg.eigvals(J)
    return float(np.abs(ev).max()), float(np.linalg.norm(J, 2)), float(np.min(np.abs(1 - ev)))

grid = []
for eta in [0.45, 0.6, 0.8, 1.0]:
    for eps in [0.05, 0.02, 0.01]:
        S = FDSolver(Params(eta=eta, eps=eps), 41, 120)
        sols = []; its = []
        for q0 in starts(S.nt):
            q, h = S.solve(q_init=q0, damping=0.3, tol=1e-10, maxit=4000)
            sols.append(q); its.append(len(h))
        spread = max(np.abs(s - sols[3]).max() for s in sols)
        rho, nrm, gap = jacobian_radius(S, sols[3])
        r = report(S, sols[3])
        row = dict(eta=eta, eps=eps, n_starts=len(sols), spread=float(spread), max_iters=int(max(its)),
                   converged=bool(max(its) < 4000), rho_DPhi=rho, norm2_DPhi=nrm, min_abs_1_minus_ev=gap,
                   J_eq=float(r['J_eq']), ratio=float(r['ratio']), max_dev=float(r['max_dev']))
        grid.append(row); print(row)
out['multistart_jacobian'] = grid
json.dump(out, open('uniqueness.json', 'w'), indent=1, default=float)
