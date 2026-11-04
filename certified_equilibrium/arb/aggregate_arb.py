"""Final equilibrium-distance bound from Arb-only inputs (Theorem: ||qhat-q*||_w <= r/(1-c)), w(t)=1+1.99t-0.68t^2."""
from decimal import Decimal as D, getcontext, ROUND_CEILING
getcontext().prec=40; getcontext().rounding=ROUND_CEILING
lam=D('2.25'); h=D(1)/D(6000); alpha=lam*h; eps=D('0.05'); T=D(1)
A=D('0.0029'); sqrtA=A.sqrt()                       # ceiling-rounded sqrt
C1=D('1.1'); C2=D('1.42')                            # sup (sqrt t + 1.5 t^1.5)/w2 and sup (1+2.25t)/w2
inp={'0.50':dict(dH=D('1.3842338e-5'),Lp=D('0.17366125'),Lt=D('0.5750040'),rnode=D('0.0021403859'),c=D('0.8053929')),
     '0.60':dict(dH=D('1.3960268e-5'),Lp=D('0.17443546'),Lt=D('0.5765457'),rnode=D('0.0021460490'),c=D('0.9692971'))}
for eta,v in inp.items():
    tail=6000*alpha**3/6; freeze=lam*v['Lt']*h/4; within=(2*lam+v['Lt'])*h
    rref=v['rnode']+tail+freeze+within
    belief=C1*v['Lp']*sqrtA; pol=C2*T*v['dH']/eps
    r=rref+belief+pol; err=r/(1-v['c'])
    print(eta,'r_ref',round(rref,7),'belief',round(belief,7),'policy',round(pol,7),'r',round(r,7),'c',v['c'],'bound',round(err,5))
