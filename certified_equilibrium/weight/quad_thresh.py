import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
import sys; sys.path.insert(0,_os.path.join(HERE,'..','..','code'))
import cert_interval as ci
src=open(_os.path.join(HERE,'..','..','code','cert_interval.py')).read()
i=src.index('def sharp_kappa'); j=src.index('def analytic_kappa_v1')
code=src[i:j].replace("w = [1 + al * k * dt for k in range(nt + 1)]","w = [1 + A('1.99') * (k * dt) - A('0.68') * (k * dt) ** 2 for k in range(nt + 1)]").replace("def sharp_kappa","def quad_kappa")
ns=dict(ci.__dict__); exec(code,ns)
def best(eta,nt=4000):
    b=None
    for th in ('10','12','14','16','18'):
        k=ns['quad_kappa'](1,2.25,'0.05',eta,th,nt=nt)['kappa']
        if b is None or float(k.upper())<b[0]: b=(float(k.upper()),th)
    return b
for eta in ():
    print(eta,best(eta),flush=True)
lo,hi=0.61,0.62
for _ in range(4):
    mid=round((lo+hi)/2,5); r=best(str(mid))
    if r[0]<1: lo=mid
    else: hi=mid
print('threshold',lo,hi)
