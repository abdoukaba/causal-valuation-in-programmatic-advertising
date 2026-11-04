import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
"""Interval-valued reference-flow validation for frozen-prior policies.
State (S,b) has 10 entries. Uses exactly evaluated coefficients, candidate
q piecewise constant, uniformization truncated after 2 events per short step.
"""
import json,math,time,sys
from mpmath import iv
from decimal import Decimal as D
from iv_hjb import Cert,I,cheb
iv.dps=35

def upper(x):
 return math.nextafter(float(x.b),math.inf)
def endpointstr(x,which):
 s=str(x.a if which=='a' else x.b)
 return s.strip('[]').split(',')[0].strip()
def tostr(x):return [repr(math.nextafter(float(x.a),-math.inf)),repr(math.nextafter(float(x.b),math.inf))]
def fromstr(pair):return iv.mpf(pair)

class Forward:
 def __init__(self,path):
  self.c=Cert(path);self.m=[[I(0) for b in range(5)] for s in range(2)]
  for s in range(2):self.m[s][4]=I('.5')
  self.Nsub=100;self.dt=I(1)/I(6000);self.alpha=self.c.lam*self.dt
  self.atten=iv.exp(-self.alpha)
  self.max_res=0;self.arg=None;self.discr=[]
 def policy(self,k,u):
  c=self.c;p=I('.5');co=[c.coeff(k,b,u) for b in range(5)]
  V=[cheb(cs,p) for cs in co]
  _,ell,g=c.data[k];r=[]
  for b in range(5):
   e=[I(1)]
   for a in range(1,min(b,2)+1):
    w=ell[0][a]+p*(ell[1][a]-ell[0][a]);gg=g[0][a]+p*(g[1][a]-g[0][a])
    win=p*ell[1][a]/w;loss=p*(1-ell[1][a])/(1-w)
    adv=gg+w*cheb(co[b-a],win)+(1-w)*cheb(co[b],loss)-V[b]
    e.append(iv.exp(adv/c.eps))
   total=sum(e,I(0));r.append([e[a]/total if a<len(e) else I(0) for a in range(3)])
  return r
 def K(self,m,pol,win):
  nm=[[I(0) for b in range(5)] for s in range(2)]
  for s in range(2):
   for b in range(5):
    p1=pol[b][1]*win[s][1] if b>=1 else I(0)
    p2=pol[b][2]*win[s][2] if b>=2 else I(0)
    nm[s][b]+=m[s][b]*(I(1)-p1-p2)
    if b>=1:nm[s][b-1]+=m[s][b]*p1
    if b>=2:nm[s][b-2]+=m[s][b]*p2
  return nm
 def step(self,k,n):
  u=I(n)/I(self.Nsub)
  pol=self.policy(k,u)
  qref=[sum((self.m[s][b]*pol[b][a] for s in range(2) for b in range(5)),I(0)) for a in range(3)]
  qhat=self.c.q[k]+[I(1)-self.c.q[k][0]-self.c.q[k][1]]
  err=sum((abs(qref[a]-qhat[a]) for a in range(3)),I(0))
  t=I(k)/I(60)+I(n)/I(6000)
  ub=upper(err/(I(1)+t))
  if ub>self.max_res:self.max_res=ub;self.arg=(k,n,ub)
  # frozen midpoint policy exact uniformization truncated at two auctions
  midpoint=I(2*n+1)/I(200)
  pmid=self.policy(k,midpoint)
  win=self.c.data[k][1]
  x=self.K(self.m,pmid,win);y=self.K(x,pmid,win)
  self.m=[[self.atten*(self.m[s][b]+self.alpha*x[s][b]+self.alpha*self.alpha*y[s][b]/2) for b in range(5)] for s in range(2)]
  return ub
 def serial(self):return [[tostr(x) for x in row] for row in self.m]
 def restore(self,arr):self.m=[[fromstr(x) for x in row] for row in arr]
 def do(self,start,end,out):
  begin=time.monotonic()
  for k in range(start,end):
   top=0
   for n in range(100):top=max(top,self.step(k,n))
   print(k,'cellmax',top,'global',self.max_res,'sec',round(time.monotonic()-begin,2),flush=True)
  summ={'start':start,'end':end,'mass':self.serial(),'max_res':self.max_res,'arg':self.arg}
  open(out,'w').write(json.dumps(summ,indent=2));print('SAVED',out)

if __name__=='__main__':
 path=sys.argv[1];start=int(sys.argv[2]);end=int(sys.argv[3]);out=sys.argv[4]
 x=Forward(path)
 if start>0:
  etatag='eta60' if '0.60' in path else 'eta50'
  prev=_os.path.join(HERE,'outputs',f'forward_{etatag}_{start-10}_{start}.json')
  prev=json.load(open(prev));x.restore(prev['mass']);x.max_res=prev['max_res'];x.arg=prev['arg']
 x.do(start,end,out)
