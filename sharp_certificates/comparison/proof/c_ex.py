"""Arb enclosure of the existence constant c_ex of Proposition 7 (Appendix C of the article).
Usage: ARB_BITS=224 python c_ex.py archived 0.45 [eps]
c_ex = eta/(4 eps) * max_{s, i<j, a!=a'} { |E^ij_sa - E^ij_sa'| + B_s (|D^ij_sa| + |D^ij_sa'|) },
B_s = U_s - L_s, L_s = lam * min_{a,x} gbar^s_a(x), U_s = lam*eps*log sum_a exp(max_x gbar^s_a(x)/eps),
x over the vertices of the endogenous simplex; abstention has w = g = 0. Horizon T = 1, lam = 2.25."""
import sys, json
from itertools import combinations
from enclose import stored, A, mx, mn, ctx
name = sys.argv[1]; eta = A(sys.argv[2]); eps = A(sys.argv[3] if len(sys.argv) > 3 else '.05'); lam = A('2.25')
M, q, z, d, D = stored(name)
acts = [None, M[1], M[2]]
def w(s, a, m): return A(0) if a is None else 1 / (1 + (-(a - m - d[s]) / z).exp())
def g(s, a, mi): return A(0) if acts[a] is None else w(s, acts[a], M[mi]) * (D[s][mi] - acts[a])
verts = [[(1 - eta) * x + (eta if (j == k) else 0) for j, x in enumerate(q)] for k in range(3)]
cvals = []
for s in (0, 1):
    gbar = [[sum((v[m] * g(s, a, m) for m in range(len(M))), A(0)) for v in verts] for a in range(3)]
    L = lam * mn(x for row in gbar for x in row)
    U = lam * eps * sum(((mx(row) / eps).exp() for row in gbar), A(0)).log()
    B = U - L
    for i, j in combinations(range(3), 2):
        for a in range(3):
            for b in range(3):
                if a == b: continue
                Ea = g(s, a, i) - g(s, a, j); Eb = g(s, b, i) - g(s, b, j)
                Da = (A(0) if acts[a] is None else w(s, acts[a], M[i]) - w(s, acts[a], M[j]))
                Db = (A(0) if acts[b] is None else w(s, acts[b], M[i]) - w(s, acts[b], M[j]))
                cvals.append(abs(Ea - Eb) + B * (abs(Da) + abs(Db)))
c = eta / (4 * eps) * mx(cvals)
print(json.dumps(dict(model=name, eta=str(sys.argv[2]), eps=str(eps), precision_bits=ctx.prec, c_ex=c.str(15), upper=float(c.upper()), below_one=bool(c < 1))))
