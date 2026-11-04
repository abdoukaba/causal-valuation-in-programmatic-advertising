"""Bounds of Theorem (time term H*sqrt(dt/2)+J*dt/2) (convergence of discretizations) with w2(t)=1+1.99t-0.68t^2, theta=14, upward decimal rounding."""
import json, os
from decimal import Decimal as D, getcontext, ROUND_CEILING
getcontext().prec = 30; getcontext().rounding = ROUND_CEILING
cons = {c['eta']: c for c in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'approx_consts.json')))}
kap = {'0.5': D('0.8053929'), '0.6': D('0.9692971'), '0.61': D('0.98575')}
rows = []
for eta, k in kap.items():
    c = cons[eta]; H = D(str(c['H'])); J = D(str(c['J'])); Cp = D(str(c['Cp']))
    r = dict(eta=eta, kappa=str(k))
    for n in (60, 1000, 4000):
        Dl = D(1)/D(n); r[f'time_n{n}'] = str(((H*(Dl/2).sqrt() + J*Dl/2)/(1-k)).quantize(D('0.0001')))
    for n, h in ((4000, D('0.001')), (4000, D('0.0001'))):
        Dl = D(1)/D(n); r[f'semi_n{n}_h{h}'] = str(((Cp*h + H*(Dl/2).sqrt() + J*Dl/2)/(1-k)).quantize(D('0.0001')))
    rows.append(r); print(r)
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'approx_bounds.json'), 'w'), indent=1)
