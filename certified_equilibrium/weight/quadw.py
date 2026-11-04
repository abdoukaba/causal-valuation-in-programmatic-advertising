import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
import sys; sys.path.insert(0,_os.path.join(HERE,'..','..','code'))
import cert_interval as ci
from flint import arb
src=open(_os.path.join(HERE,'..','..','code','cert_interval.py')).read()
# reuse sharp_kappa with the linear weight replaced by w(t)=1+1.99t-0.68t^2 (increasing on [0,1])
i=src.index('def sharp_kappa'); j=src.index('def analytic_kappa_v1')
code=src[i:j].replace("w = [1 + al * k * dt for k in range(nt + 1)]","w = [1 + A('1.99') * (k * dt) - A('0.68') * (k * dt) ** 2 for k in range(nt + 1)]").replace("def sharp_kappa","def quad_kappa")
ns=dict(ci.__dict__); exec(code,ns)
for eta in ('0.5','0.6'):
    for th in ('14','16'):
        r=ns['quad_kappa'](1,2.25,'0.05',eta,th,nt=4000,alpha='0')
        print(eta,th,r['kappa'].upper().str(10,radius=False),flush=True)
