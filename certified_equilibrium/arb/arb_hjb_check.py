import sys, json
from flint import arb
from iv_taylor import C2, D2, O, box
def ub(x): return x.abs_upper()
def upper(c,k,pc,rp,uc,ru):
    mid=c.residualJ(k,D2(str(pc),p=O),D2(str(uc),u=O))
    big=c.residualJ(k,D2(box(pc-rp,pc+rp),p=O),D2(box(uc-ru,uc+ru),u=O))
    P=arb(str(rp)); U=arb(str(ru)); best=None
    for r,b in zip(mid,big):
        v=ub(r.v)+P*ub(r.p)+U*ub(r.u)+P*P*ub(b.pp)/2+P*U*ub(b.pu)+U*U*ub(b.uu)/2
        v=v.upper()
        best=v if best is None else best.max(v)
    return best
from decimal import Decimal as D
path=sys.argv[1]; c=C2(path); top=None; rows=[]
for k in range(60):
    for j in range(10):
        pc=D(2*j+1)/D(20)
        v=upper(c,k,pc,D('0.05'),D('0.5'),D('0.5')); rows.append((k,j,float(v.upper())))
        top=v if top is None else top.max(v)
print(path,'max HJB residual (Arb upper):',top.upper().str(12,radius=False))
json.dump(rows,open(path.replace('.json','_arbhjb.json'),'w'))
