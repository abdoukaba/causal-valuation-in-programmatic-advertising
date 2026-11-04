"""Common-rival comparison in the data-calibrated market (Criteo offsets, eta=0.45, eps=0.05, 21x60 grid).
All single-bidder policies face the SAME rival population: the Shannon (logit) equilibrium flow q_S.
Reports monetary profit at the initial state of: the Shannon equilibrium policy, the KL-anchored Gibbs policy
(0.9 on the data-selected bid h) best-responding to q_S, the deterministic rule 'always bid h when feasible',
'always bid 2h when feasible', and the monetary best response to q_S. Also the same under the anchored
equilibrium flow q_A. Writes ../results/common_rival.json."""
import json, numpy as np, calib_full as cf
P = cf.primitives()
rho = np.full(3, .05); rho[1] = .90
def run(qflow_from):
    mS = cf.model(P); sS = mS.solve(0, max_iter=1500, damping=.4)
    mA = cf.model(P, rho=rho); sA = mA.solve(0, max_iter=1500, damping=.4)
    q = sS['q'] if qflow_from == 'shannon' else sA['q']
    out = {}
    _, polS = mS.backward(q); _, polA = mA.backward(q)
    out['shannon_policy'] = mS.monetary(q, polS)
    out['anchored_policy'] = mS.monetary(q, polA)
    for name, k in (('always_h', 1), ('always_2h', 2)):
        pol = np.zeros_like(polS)
        for b in range(mS.nb):
            j = min(k, b) if mS.actions[min(k, b)] <= mS.budgets[b] + 1e-12 else 0
            # feasible bids: index i feasible iff actions[i] <= budget b
            feas = [i for i in range(3) if mS.actions[i] <= mS.budgets[b] + 1e-12]
            jj = k if k in feas else max(feas)
            pol[:, b, :, jj] = 1.0
        out[name] = mS.monetary(q, pol)
    out['best_response'] = mS.monetary(q)
    return out
res = {'against_shannon_equilibrium': run('shannon'), 'against_anchored_equilibrium': run('anchored')}
print(json.dumps(res, indent=1))
json.dump(res, open('../results/common_rival.json', 'w'), indent=1)
