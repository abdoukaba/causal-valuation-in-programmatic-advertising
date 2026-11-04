"""Exact-rational primitive bounds + outward decimal, whole-time-cell certificate.

For fixed smooth scoring benchmark only (A0={0,.05,.10}, baseline Table 1).
Proof of upper bound in MOR_TIME_WEIGHTED_THEOREM.tex.

- Logistic probabilities: enclosed using Taylor sums and explicit positive-series tails
  in fractions.Fraction, no floating point.
- log(3): enclosed using atanh series with geometric tail in fractions.Fraction.
- All positive time-cell propagation: Decimal with ROUND_CEILING at 40 digits.
- Every denominator 1-c*dt is computed with ROUND_FLOOR to be conservative.
- Exponential growth uses exact inequality exp(x) <= (1-x)^(-1), 0<=x<1.
- Coefficients converted to lower/upper 12-decimal bounds from exact Fractions.
The computed gain is a mathematically conservative upper bound on the continuum
response-map weighted Lipschitz constant under the stated model and estimates.
"""
from __future__ import annotations
from fractions import Fraction as F
from decimal import Decimal as D, localcontext, Context, ROUND_CEILING, ROUND_FLOOR
import json, hashlib, argparse, pathlib

class IV:
 def __init__(self,lo,hi=None):
  self.lo=F(lo); self.hi=F(lo if hi is None else hi)
  if self.lo>self.hi:raise ValueError('inverted')
 def __add__(self,x):
  x=iv(x);return IV(self.lo+x.lo,self.hi+x.hi)
 __radd__=__add__
 def __neg__(self):return IV(-self.hi,-self.lo)
 def __sub__(self,x):return self+-iv(x)
 def __rsub__(self,x):return iv(x)+-self
 def __mul__(self,x):
  x=iv(x);z=[a*b for a in (self.lo,self.hi) for b in (x.lo,x.hi)];return IV(min(z),max(z))
 __rmul__=__mul__
 def __truediv__(self,x):
  x=iv(x)
  if x.lo<=0<=x.hi: raise ZeroDivisionError()
  return self*IV(1/x.hi,1/x.lo)
 def abshi(self):return max(abs(self.lo),abs(self.hi))
def iv(x):return x if isinstance(x,IV) else IV(F(x))

def exp_positive(y,terms=60):
 assert y>=0 and y<F(3)
 total=F(1); term=F(1)
 for k in range(1,terms+1):
  term=term*y/k;total+=term
 nextterm=term*y/(terms+1)
 remainder=nextterm/(1-y/F(terms+2))
 return IV(total,total+remainder)
def exp_interval(x):
 x=F(x)
 if x>=0:return exp_positive(x)
 pos=exp_positive(-x);return IV(1/pos.hi,1/pos.lo)
def logistic(x):
 e=exp_interval(-x)
 return IV(1/(1+e.hi),1/(1+e.lo))
def ln3_upper(n=100):
 z=F(1,2); total=F(0)
 for j in range(n):total+=2*z**(2*j+1)/F(2*j+1)
 rem=2*z**(2*n+1)/(F(2*n+1)*(1-z*z))
 return total+rem

def quantize_fraction(x,digits=12,upper=True):
 unit=10**digits
 n=x.numerator*unit//x.denominator
 if upper and F(n,unit)<x:n+=1
 return D(n)/D(unit)

def model_constants(eta:F):
 bids=[F(0),F(1,20),F(1,10)]
 q0=[F(1,5),F(2,5),F(2,5)]
 d=[-F(1,100),F(1,100)]
 noise=F(1,5)
 ws={}; gs={}
 for s in range(2):
  for a in (1,2):
   for mi,m in enumerate(bids):
    ww=logistic((bids[a]-m-d[s])/noise)
    value=(F(60,1000) if s==0 else F(95,1000))+F(3,100)*m
    ws[s,a,mi]=ww;gs[s,a,mi]=ww*(value-bids[a])
 vertices=[[(1-eta)*q0[i]+(eta if i==k else 0) for i in range(3)] for k in range(3)]
 prim={}; lp_hi=F(1);r_hi=F(0)
 for a in (1,2):
  for s in range(2):
   for mi in range(3):r_hi=max(r_hi,gs[s,a,mi].hi)
  ell=[ [sum((ws[s,a,mi]*Q[mi] for mi in range(3)), IV(0)) for Q in vertices] for s in range(2)]
  eslo=[min(v.lo for v in ell[s]) for s in (0,1)]
  eshi=[max(v.hi for v in ell[s]) for s in (0,1)]
  ellmin=min(eslo);ellmax=max(eshi)
  gamma=max(max(gs[s,a,mi].hi for mi in range(3))-min(gs[s,a,mi].lo for mi in range(3)) for s in (0,1))
  omega=max(max(ws[s,a,mi].hi for mi in range(3))-min(ws[s,a,mi].lo for mi in range(3)) for s in (0,1))
  Lg=max((gs[1,a,mi]-gs[0,a,mi]).abshi() for mi in range(3))
  Lw=max((ws[1,a,mi]-ws[0,a,mi]).abshi() for mi in range(3))
  for mi in range(3):
   for s in (0,1):
    ss=1-s
    lp_hi=max(lp_hi,(ws[s,a,mi]/ws[ss,a,mi]).hi,((1-ws[s,a,mi])/(1-ws[ss,a,mi])).hi)
  O1=O0=F(0)
  l0=IV(eslo[0],eshi[0]);l1=IV(eslo[1],eshi[1])
  for i in range(3):
   for j in range(3):
    w1=ws[1,a,i]-ws[1,a,j];w0=ws[0,a,i]-ws[0,a,j]
    O1=max(O1,(l0*w1-l1*w0).abshi())
    O0=max(O0,((1-l0)*(-w1)-(1-l1)*(-w0)).abshi())
  chi=O1/(4*ellmin)+O0/(4*(1-ellmax))
  if not(ellmin*(1-ellmin)>F(1,8) and ellmax*(1-ellmax)>F(1,8)):
   raise ValueError('Interior win condition not certified')
  prim[a]={'gamma':gamma,'omega':omega,'Lg':Lg,'Lw':Lw,'chi':chi,'ellmin':ellmin,'ellmax':ellmax}
  
 return prim,lp_hi,r_hi,ln3_upper()

UP=Context(prec=38,rounding=ROUND_CEILING)
DOWN=Context(prec=38,rounding=ROUND_FLOOR)
def up_add(*items):
 with localcontext(UP):
  z=D(0)
  for it in items:z+=it
  return +z
def up_mul(*items):
 with localcontext(UP):
  z=D(1)
  for it in items:z*=it
  return +z
def up_div(x,y):
 with localcontext(UP):return x/y
def lower_denom_one_minus(x):
 with localcontext(DOWN):return D(1)-x

def certificate(eta='0.55',epsilon='0.05',theta='16',cells=1200):
 lam=D('2.25');T=D(1);eta_d=D(eta);eps=D(epsilon);th=D(theta)
 prim_q, lp_q, r_q, ln3_q=model_constants(F(eta))
 pc={a:{k:quantize_fraction(v,12,upper=(k!='ellmin')) for k,v in c.items()} for a,c in prim_q.items()}
 lp=quantize_fraction(lp_q);r=quantize_fraction(r_q);ln3=quantize_fraction(ln3_q)
 dt=up_div(T,D(cells));t=[up_mul(D(i),dt) for i in range(cells+1)]
 vb=[up_mul(lam,(T-s),up_add(r,up_mul(eps,ln3))) for s in t]
 maxlw=max(pc[a]['Lw'] for a in (1,2))
 gL=up_mul(lam,up_add((lp-D(1)),up_mul(D(2),maxlw,maxlw)))
 L=[D(0)]*(cells+1)
 for i in range(cells-1,-1,-1):
  source=max(up_add(pc[a]['Lg'],up_mul(pc[a]['Lw'],vb[i])) for a in (1,2))
  # Explicit Euler rational supersolution for -L'=gL L+lambda S0(t), S0 <= S0(t_i).
  gdt=up_mul(gL,dt)
  den=lower_denom_one_minus(gdt)
  if den<=0:raise ValueError('Time cell too coarse: gL*dt>=1')
  L[i]=up_div(up_add(L[i+1],up_mul(lam,source,dt)),den)
 D_a={a:[up_mul(eta_d,D('.5'),up_add(c['gamma'],up_mul(c['omega'],up_add(vb[i],up_mul(D(2),L[i],c['Lw']))),up_mul(L[i],c['chi']))) for i in range(cells+1)] for a,c in pc.items()}
 Db=[[D(0)]*(cells+1) for _ in range(5)]
 for b in range(1,5):
  for i in range(cells+1):Db[b][i]=max(D_a[a][i] for a in (1,2) if a<=b)
 Lambda=[max(up_div(up_add(c['Lg'],up_mul(c['Lw'],up_add(vb[i],up_mul(D(2),L[i],c['Lw']))),up_mul(up_add(lp,D(1)),L[i])),eps) for a,c in pc.items()) for i in range(cells+1)]
 k=max(up_mul(eta_d,up_add(c['omega'],up_mul(th,c['chi'],D('.5')))) for c in pc.values())
 c=[up_mul(lam,up_add(lp-D(1),up_div(up_add(up_mul(D(2),maxlw),Lambda[i]),th))) for i in range(cells+1)]
 # Exact rational quadratic weight w(t)=1+1.99t-0.68t^2, increasing on [0,1].
 aW=F(199,100);bW=-F(68,100)
 wfrac=[F(1)+aW*F(i,cells)+bW*F(i,cells)**2 for i in range(cells+1)]
 assert all(wfrac[i+1]>wfrac[i]>0 for i in range(cells))
 wup=[quantize_fraction(x,20,upper=True) for x in wfrac]
 wlo=[quantize_fraction(x,20,upper=False) for x in wfrac]
 E=[[D(0)]*(cells+1) for _ in range(5)]
 for i in range(cells-1,-1,-1):
  for b in range(1,5):
   E[b][i]=up_add(E[b][i+1],up_mul(lam,dt,Db[b][i],wup[i+1]))
 P=[D(0)]*cells
 for i in range(cells):
  P[i]=max(up_div(up_add(up_mul(D_a[a][i],wup[i+1]),up_mul(pc[a]['ellmin'],E[b-a][i]),up_mul(D(2)-pc[a]['ellmin'],E[b][i])),eps) for b in range(1,5) for a in (1,2) if a<=b)
 Fupper=[D(0)]*(cells+1)
 ratios=[]
 for i in range(cells):
  z=up_add(P[i],up_mul(k,wup[i+1]))
  cdt=up_mul(c[i],dt)
  den=lower_denom_one_minus(cdt)
  if den<=0:raise ValueError('Time cell too coarse: c*dt>=1')
  Fupper[i+1]=up_div(up_add(Fupper[i],up_mul(lam,z,dt)),den)
  fac=max(D(1),up_div(Lambda[i],th))
  out=up_add(P[i],up_mul(fac,Fupper[i+1]))
  ratios.append(up_div(out,wlo[i]))
 worst=max(ratios)
 out={'eta':eta,'epsilon':epsilon,'theta':theta,'cells':cells,
      'weighted_upper_bound':str(worst),'strictly_below_one':bool(worst<D(1)),
      'worst_cell':int(max(range(cells),key=lambda j:ratios[j])),
      'rounding':'Fraction exact-rational enclosures of exp, log3; 12-place outward constants; Decimal ROUND_CEILING at 38 digits, ROUND_FLOOR for positive denominators; exp(x)<=1/(1-x)',
      'positive_weight':'w(t)=1+1.99t-0.68t^2, increasing positive on [0,1]',
      'primitive_constants':{str(a):{k:str(v) for k,v in z.items()} for a,z in pc.items()},
      'LPsi_upper':str(lp),'rplus_upper':str(r),'ln3_upper':str(ln3),
      'max_cdt_upper':str(max(up_mul(x,dt) for x in c)),'max_gLdt_upper':str(up_mul(gL,dt)),
      'max_Lambda':str(Lambda[0]),'L0_upper':str(L[0])}
 return out

if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--eta',default='.55');pa.add_argument('--eps',default='.05');pa.add_argument('--theta',default='16');pa.add_argument('--cells',type=int,default=1200);pa.add_argument('--output',default='weighted_gain_certificate.json')
 a=pa.parse_args()
 obj=certificate(a.eta,a.eps,a.theta,a.cells);pathlib.Path(a.output).write_text(json.dumps(obj,indent=2)+'\n'); print(json.dumps({k:v for k,v in obj.items() if k not in ('primitive_constants','rounding')},indent=2))
