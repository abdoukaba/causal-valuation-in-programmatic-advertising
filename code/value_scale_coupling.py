"""(i) Monetary-scale sensitivity of the data-calibrated market: regime values Delta multiplied by s.
(ii) Coupling bounds: the model assumes the hidden regime S is independent of the rival bid M. Over ALL joint
laws of (S,M) with the calibrated marginals (prior p0, background law q0), we bound by linear programming
(a) the single-auction plug-in profit of each bid and (b) the won-impression value gap E[D|W=1]-E[D].
Writes ../results/value_scale_coupling.json. Run from results/ with PYTHONPATH=../code."""
import json, numpy as np
from scipy.optimize import linprog
from scipy.special import expit
import calib_full as cf
out = {'value_scale': {}, 'coupling': {}}
rho = np.full(3, .05); rho[1] = .90
for s in (0.5, 0.75, 1.0, 1.5):
    P = cf.primitives(scale=s)
    R1 = cf.single_auction_profit(P)
    S = cf.equilibrium(P, None, seeds=3); A = cf.equilibrium(P, rho, seeds=3)
    out['value_scale'][str(s)] = dict(plugin_profits=R1.tolist(), selected=int(np.argmax(R1)),
        shannon_eq=S['eq'], shannon_br=S['br'], anchored_eq=A['eq'], anchored_br=A['br'], spread=max(S['spread'], A['spread']))
    print('scale', s, out['value_scale'][str(s)], flush=True)
# coupling LP
P = cf.primitives(); MG = cf.MG; p = np.array([1 - P['p0'], P['p0']]); Q = P['q0']; D = P['Delta']; d = P['d']; z = P['zeta']
nS, nM = 2, len(Q)
A_eq = []; b_eq = []
for s_ in range(nS):
    row = np.zeros(nS * nM); row[s_ * nM:(s_ + 1) * nM] = 1; A_eq.append(row); b_eq.append(p[s_])
for m in range(nM):
    row = np.zeros(nS * nM); row[m::nM] = 1; A_eq.append(row); b_eq.append(Q[m])
A_eq = np.array(A_eq); b_eq = np.array(b_eq)
indep = np.outer(p, Q).ravel()
for a in (.05, .10):
    w = np.array([expit((a - MG - d[s_]) / z) for s_ in range(nS)]).ravel()
    Dv = D.ravel()
    prof = w * (Dv - a)
    lo = linprog(prof, A_eq=A_eq, b_eq=b_eq, bounds=(0, None)); hi = linprog(-prof, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
    # gap = sum(pi*Dv*w)/sum(pi*w) - sum(pi*Dv); sweep the win probability t and solve an LP at each t
    W0 = indep @ w; gaps = []
    tmin = linprog(w, A_eq=A_eq, b_eq=b_eq, bounds=(0, None)).fun; tmax = -linprog(-w, A_eq=A_eq, b_eq=b_eq, bounds=(0, None)).fun
    for t in np.linspace(tmin, tmax, 401)[1:-1]:
        Ae = np.vstack([A_eq, w]); be = np.append(b_eq, t)
        c = Dv * w / t - Dv
        r1 = linprog(-c, A_eq=Ae, b_eq=be, bounds=(0, None)); r2 = linprog(c, A_eq=Ae, b_eq=be, bounds=(0, None))
        if r1.status == 0: gaps.append(-r1.fun)
        if r2.status == 0: gaps.append(r2.fun)
    out['coupling'][str(a)] = dict(plugin_profit_independent=float(indep @ prof), plugin_profit_min=float(lo.fun), plugin_profit_max=float(-hi.fun),
        gap_independent=float(indep @ (Dv * w) / W0 - indep @ Dv), gap_min=float(min(gaps)), gap_max=float(max(gaps)),
        win_prob_independent=float(W0), win_prob_range=[float(tmin), float(tmax)])
    print('coupling', a, out['coupling'][str(a)], flush=True)
json.dump(out, open('../results/value_scale_coupling.json', 'w'), indent=1)
