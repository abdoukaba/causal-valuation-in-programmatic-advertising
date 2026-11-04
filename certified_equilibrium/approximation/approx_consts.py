"""Arb enclosures of the constants in the time-regularity and discretization results (Section 7).
alpha=D*/eps, B=(sigma/4)(L*/eps+Lambda*), A=lam(r^+ + eps log3 + Vbar*)/eps + 2 lam,
H=B/(1-alpha), J=A/(1-alpha), C_p (belief projection), with theta=Lambda*."""
import sys, json
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'code'))
from cert_interval import A, amax, primitives, ctx
from flint import arb
ctx.prec = 128
def consts(eta, T='1', lam='2.25', eps='0.05', sigma='0.20', nt=4000):
    T, lam, eps, eta_, sig = A(T), A(lam), A(eps), A(eta), A(sigma)
    pc, LPsi, rplus, _ = primitives(eta_)
    logJ = arb(3).log()
    Lw2 = amax([c['Lw']**2 for c in pc.values()]); Lwmax = amax([c['Lw'] for c in pc.values()])
    g = lam*(LPsi-1+2*Lw2)
    dt = T/nt; egdt = (g*dt).exp()
    Vbar = [lam*(T-k*dt)*(rplus+eps*logJ) for k in range(nt+1)]
    L = A(0)
    for k in range(nt-1, -1, -1):
        S0 = amax([c['Lg']+c['Lw']*Vbar[k] for c in pc.values()])
        L = L*egdt + lam*S0*(egdt-1)/g
    V0 = Vbar[0]
    Dstar = amax([eta_/2*(c['gamma']+c['omega']*(V0+2*L*c['Lw'])+L*c['beta']) for c in pc.values()])
    Lam = amax([c['Lg']+c['Lw']*(V0+2*L*c['Lw'])+(LPsi+1)*L for c in pc.values()])/eps
    alpha = Dstar/eps
    Bc = sig/4*(L/eps+Lam)
    Ac = lam*(rplus+eps*logJ+V0)/eps + 2*lam
    H = Bc/(1-alpha); J = Ac/(1-alpha)
    th = Lam
    c0 = lam*((LPsi-1)+(2*Lwmax+Lam)/th)
    dpi_coef = (1+lam*T)*L/(2*eps)            # delta_pi <= dpi_coef * h
    Cp = dpi_coef*(1+lam*T) + th/2*lam*((c0*T).exp()-1)/c0
    u = lambda x: float(x.upper())
    return dict(eta=eta, rplus=u(rplus), Vbar0=u(V0), L0=u(L), Dstar=u(Dstar), Lambda0=u(Lam), LPsi=u(LPsi),
                alpha=u(alpha), A=u(Ac), B=u(Bc), H=u(H), J=u(J), c0=u(c0), Cp=u(Cp))
out = [consts(e) for e in ('0.5', '0.6', '0.61')]
for o in out: print(json.dumps(o))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'approx_consts.json'), 'w'), indent=1)
