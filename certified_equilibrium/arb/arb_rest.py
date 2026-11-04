"""Arb recomputation of the policy belief-Lipschitz bound L_p (600 boxes), the prior time-Lipschitz bound L_t,
and the ten-state forward reference residual r_node, for one candidate file.  Same formulas as the v8 scripts;
all arithmetic in Arb balls; every reported number is an Arb upper end."""
import sys, json
from flint import arb
from arbshim import iv
from iv_taylor import C2, D2, O, box
from iv_policy import pol_jets
from decimal import Decimal as D
path = sys.argv[1]; c = C2(path)
def amax(xs):
    m = xs[0]
    for x in xs[1:]: m = m.max(x)
    return m
U = lambda x: x.abs_upper()
# --- L_p
Lp = None
for k in range(60):
    for j in range(10):
        pc = D(2*j+1)/D(20); rp, uc, ru = D('0.05'), D('0.5'), D('0.5')
        cen = pol_jets(c, k, D2(str(pc), p=O), D2(str(uc), u=O))
        out = pol_jets(c, k, D2(box(pc-rp, pc+rp), p=O), D2(box(uc-ru, uc+ru), u=O))
        R, Ru = arb(str(rp)), arb(str(ru)); i = 0; per = []
        for cnt in (2, 3, 3, 3):
            per.append(sum((U(x.p) + R*U(y.pp) + Ru*U(y.pu) for x, y in zip(cen[i:i+cnt], out[i:i+cnt])), arb(0)).upper()); i += cnt
        v = amax(per); Lp = v if Lp is None else Lp.max(v)
print('L_p', Lp.upper().str(10, radius=False), flush=True)
# --- L_t at the prior
Lt = None
for k in range(60):
    jets = pol_jets(c, k, D2('.5'), D2(box(D(0), D(1)), u=O)); i = 0; per = []
    for cnt in (2, 3, 3, 3):
        per.append((60*sum((U(x.u) for x in jets[i:i+cnt]), arb(0))).upper()); i += cnt
    v = amax(per); Lt = v if Lt is None else Lt.max(v)
print('L_t', Lt.upper().str(10, radius=False), flush=True)
# --- ten-state forward reference
from iv_hjb import cheb
I = lambda v: arb(str(v))
m = [[I(0) for b in range(5)] for s in range(2)]
for s in range(2): m[s][4] = I('0.5')
dt = I(1)/I(6000); alpha = c.lam*dt; atten = (-alpha).exp()
def policy(k, u):
    p = I('0.5'); co = [c.coeff(k, b, u) for b in range(5)]; V = [cheb(cs, p) for cs in co]
    _, ell, g = c.data[k]; r = []
    for b in range(5):
        e = [I(1)]
        for a in range(1, min(b, 2)+1):
            w = ell[0][a]+p*(ell[1][a]-ell[0][a]); gg = g[0][a]+p*(g[1][a]-g[0][a])
            win = p*ell[1][a]/w; loss = p*(1-ell[1][a])/(1-w)
            adv = gg+w*cheb(co[b-a], win)+(1-w)*cheb(co[b], loss)-V[b]
            e.append((adv/c.eps).exp())
        tot = sum(e, I(0)); r.append([e[a]/tot if a < len(e) else I(0) for a in range(3)])
    return r
def K(mm, pol, win):
    nm = [[I(0) for b in range(5)] for s in range(2)]
    for s in range(2):
        for b in range(5):
            p1 = pol[b][1]*win[s][1] if b >= 1 else I(0); p2 = pol[b][2]*win[s][2] if b >= 2 else I(0)
            nm[s][b] += mm[s][b]*(1-p1-p2)
            if b >= 1: nm[s][b-1] += mm[s][b]*p1
            if b >= 2: nm[s][b-2] += mm[s][b]*p2
    return nm
rmax = None
for k in range(60):
    win = c.data[k][1]; qhat = c.q[k]+[I(1)-c.q[k][0]-c.q[k][1]]
    for n in range(100):
        pol = policy(k, I(n)/I(100))
        qref = [sum((m[s][b]*pol[b][a] for s in range(2) for b in range(5)), I(0)) for a in range(3)]
        t = I(k)/I(60)+I(n)/I(6000)
        err = (sum((abs(qref[a]-qhat[a]) for a in range(3)), I(0))/(1+t)).upper()
        rmax = err if rmax is None else rmax.max(err)
        pm = policy(k, I(2*n+1)/I(200)); x = K(m, pm, win); y = K(x, pm, win)
        m = [[atten*(m[s][b]+alpha*x[s][b]+alpha*alpha*y[s][b]/2) for b in range(5)] for s in range(2)]
    if k % 10 == 9: print('forward k', k, 'r_node so far', rmax.upper().str(8, radius=False), 'mass rad', max(float(m[s][b].rad()) for s in range(2) for b in range(5)), flush=True)
print('r_node', rmax.upper().str(10, radius=False), flush=True)
