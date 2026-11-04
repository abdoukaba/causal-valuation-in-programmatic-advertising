"""Independent symbolic checks of the two new perspective identities."""
import sympy as s,json
from pathlib import Path
p,l0,l1,d0,d1=s.symbols('p l0 l1 d0 d1',real=True)
F,G,Fp,Gp=s.symbols('F G Fp Gp',real=True)
w=(1-p)*l0+p*l1;u=p*l1/w;v=p*(1-l1)/(1-w)
a0F=F-u*Fp;a1F=F+(1-u)*Fp
a0G=G-v*Gp;a1G=G+(1-v)*Gp
actual=s.diff(w,p)*(F-G)+w*Fp*s.diff(u,p)+(1-w)*Gp*s.diff(v,p)
target=l1*a1F+(1-l1)*a1G-l0*a0F-(1-l0)*a0G
actualq=((1-p)*d0+p*d1)*(F-G)+w*Fp*(s.diff(u,l0)*d0+s.diff(u,l1)*d1)+(1-w)*Gp*(s.diff(v,l0)*d0+s.diff(v,l1)*d1)
targetq=(1-p)*d0*(a0F-a0G)+p*d1*(a1F-a1G)
r={'belief_perspective_identity':str(s.factor(actual-target)),
   'rival_perspective_identity':str(s.factor(actualq-targetq))}
assert all(x=='0' for x in r.values())
Path(__file__).with_name('symbolic_identity_checks.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
