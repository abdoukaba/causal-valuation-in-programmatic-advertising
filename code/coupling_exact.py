"""Exact bounds over ALL couplings of (regime S, rival bid M) with the calibrated marginals, for the data-calibrated
market (Criteo offsets). Replaces the grid search of value_scale_coupling.py for the selection gap.

The feasible set is the 2x4 transportation polytope Pi (marginals p, q0). The plug-in profit and the win
probability are linear in pi, so their extrema are attained at vertices. The selection gap
    g(pi) = <pi, D*w> / <pi, w> - <pi, D>
is not linear. For a fixed win probability t=<pi,w>, g is linear on the slice Pi ∩ {<pi,w>=t}, so its extrema are
attained at vertices of the slice, each of which lies on an edge of Pi. Hence sup/inf of g over Pi are attained on
segments between vertices of Pi. We enumerate all basic feasible solutions (vertices) exactly in rational arithmetic
from the binary64 inputs, and on every segment [v1,v2] maximize/minimize the one-dimensional function
    g(l) = (a + b l)/(c + d l) - (e + f l),  l in [0,1],
at the endpoints and at the interior critical points (b c - a d)/(c + d l)^2 = f, solved in closed form.
Evaluation uses 60-digit mpmath arithmetic; reported bounds are rounded outward. Writes ../results/coupling_exact.json.
Run from results/ with PYTHONPATH=../code."""
import json, itertools
from fractions import Fraction as F
import mpmath as mp
import numpy as np
from scipy.special import expit
import calib_full as cf
mp.mp.dps = 60
P = cf.primitives(); MG = cf.MG
p = [F(1) - F(float(P['p0'])), F(float(P['p0']))]
q = [F(float(x)) for x in P['q0']]; q[-1] = F(1) - sum(q[:-1])   # exact marginals summing to one
nS, nM = 2, len(q)
idx = [(s, m) for s in range(nS) for m in range(nM)]
# equality constraints (drop one redundant row)
rows = []
for s in range(nS): rows.append(([1 if i[0] == s else 0 for i in idx], p[s]))
for m in range(nM - 1): rows.append(([1 if i[1] == m else 0 for i in idx], q[m]))
def solve(basis):
    A = [[F(r[0][j]) for j in basis] for r in rows]; b = [r[1] for r in rows]; n = len(basis)
    M = [A[i] + [b[i]] for i in range(n)]
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if piv is None: return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c] / M[c][c]; M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    x = [M[i][n] / M[i][i] for i in range(n)]
    if any(v < 0 for v in x): return None
    v = [F(0)] * len(idx)
    for j, val in zip(basis, x): v[j] = val
    return tuple(v)
verts = sorted({v for B in itertools.combinations(range(len(idx)), len(rows)) if (v := solve(B)) is not None})
out = dict(n_vertices=len(verts), method=__doc__.split('\n')[1])
for a in (0.05, 0.10):
    w = [F(float(expit((a - MG[m] - P['d'][s]) / P['zeta']))) for s, m in idx]
    D = [F(float(P['Delta'][s][m])) for s, m in idx]
    lin = lambda v, c: sum(x * y for x, y in zip(v, c))
    prof = [wi * (Di - F(a).limit_denominator(1000)) for wi, Di in zip(w, D)]
    pv = [lin(v, prof) for v in verts]; wv = [lin(v, w) for v in verts]
    def g_seg(v1, v2):
        A1, A2 = lin(v1, [x * y for x, y in zip(D, w)]), lin(v2, [x * y for x, y in zip(D, w)])
        C1, C2 = lin(v1, w), lin(v2, w); E1, E2 = lin(v1, D), lin(v2, D)
        tomp = lambda x: mp.mpf(x.numerator) / x.denominator
        a_, b_, c_, d_, e_, f_ = map(tomp, (A1, A2 - A1, C1, C2 - C1, E1, E2 - E1))
        g = lambda l: (a_ + b_ * l) / (c_ + d_ * l) - (e_ + f_ * l)
        cand = [mp.mpf(0), mp.mpf(1)]
        num = b_ * c_ - a_ * d_
        if d_ != 0 and f_ != 0 and num / f_ > 0:
            for sgn in (1, -1):
                l = (sgn * mp.sqrt(num / f_) - c_) / d_
                if 0 < l < 1: cand.append(l)
        vals = [g(l) for l in cand]
        return min(vals), max(vals)
    lo, hi = mp.inf, -mp.inf
    for i in range(len(verts)):
        for j in range(i, len(verts)):
            a1, a2 = g_seg(verts[i], verts[j]); lo = min(lo, a1); hi = max(hi, a2)
    indep = tuple(p[s] * q[m] for s, m in idx)
    g0 = lin(indep, [x * y for x, y in zip(D, w)]) / lin(indep, w) - lin(indep, D)
    out[str(a)] = dict(profit_min=float(min(pv)), profit_max=float(max(pv)), profit_independent=float(lin(indep, prof)),
                       win_prob_min=float(min(wv)), win_prob_max=float(max(wv)),
                       gap_min=mp.nstr(lo, 20), gap_max=mp.nstr(hi, 20), gap_independent=float(g0),
                       gap_min_rounded_down_4=float(mp.floor(lo * 10**4) / 10**4), gap_max_rounded_up_4=float(mp.ceil(hi * 10**4) / 10**4))
    print(a, out[str(a)], flush=True)
json.dump(out, open('../results/coupling_exact.json', 'w'), indent=1)
