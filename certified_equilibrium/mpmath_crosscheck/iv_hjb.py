import json,sys,time
from mpmath import iv
iv.dps=30

def I(v):
 if isinstance(v,(str,int)):return iv.mpf(str(v))
 if isinstance(v,float):return iv.mpf(repr(v))
 return v

def box(a,b):return iv.mpf([str(a),str(b)])

# Chebyshev recurrence in interval mode; input z = 2p-1. Low-degree coefficients nearly affine.
def cheb(cs,p):
 z=2*p-1
 if len(cs)==1:return cs[0]
 t0=I(1);t1=z;out=cs[0]+cs[1]*t1
 for c in cs[2:]:
  t0,t1=t1,2*z*t1-t0
  out+=c*t1
 return out

def cheb2(cs,p):
 # exact Chebyshev second derivative via polynomial basis recursion with derivatives
 z=2*p-1
 a=I(1);ad=I(0);add=I(0)
 b=z;bd=I(1);bdd=I(0)
 out=I(0)
 if len(cs)>2:
  for c in cs[2:]:
   cval=2*z*b-a
   cder=2*b+2*z*bd-ad
   cdd=4*bd+2*z*bdd-add
   out+=c*cdd
   a,ad,add=b,bd,bdd
   b,bd,bdd=cval,cder,cdd
 return 4*out

def norm_ub(x):return max(abs(float(x.a)),abs(float(x.b)))

def show(x):return (float(x.a),float(x.b))

class Cert:
 def __init__(self,path):
  self.obj=json.load(open(path));self.n=self.obj['n'];self.eta=I(self.obj['eta'])
  self.q=[[I(z) for z in row] for row in self.obj['q']]
  self.co=[[[I(x) for x in row] for row in block] for block in self.obj['co']]
  self.d=[[[[I(x) for x in row] for row in block] for block in cell] for cell in self.obj['d']]
  self.dt=I('1')/I(60);self.sigma=I('.2');self.lam=I('2.25');self.eps=I('.05')
  self.m=[I(0),I('.05'),I('.10')];self.dlt=[I('-.01'),I('.01')]
  self.vs=[[I('.060')+I('.03')*m for m in self.m],[I('.095')+I('.03')*m for m in self.m]]
  self.ws=[[[I(1)/(I(1)+iv.exp(-(self.m[a]-self.m[j]-self.dlt[s])/I('.20'))) if a>0 else I(0) for j in range(3)] for a in range(3)] for s in range(2)]
  self.cp=[]
  for k in range(60):
   block=[]
   for b in range(5):
    byj=[]
    for j in range(self.n+1):
     y0=self.co[k][b][j];y1=self.co[k+1][b][j];d0=self.d[k][0][b][j];d1=self.d[k][1][b][j]
     byj.append((y0,self.dt*d0,-3*y0-2*self.dt*d0+3*y1-self.dt*d1,2*y0+self.dt*d0-2*y1+self.dt*d1))
    block.append(byj)
   self.cp.append(block)
  self.data=[]
  for k in range(60):
   Q=[(I(1)-self.eta)*r+self.eta*q for r,q in zip([I('.2'),I('.4'),I('.4')],self.q[k]+[I(1)-self.q[k][0]-self.q[k][1]])]
   ell=[[sum((Q[j]*self.ws[s][a][j] for j in range(3)),I(0)) for a in range(3)] for s in range(2)]
   g=[[sum((Q[j]*self.ws[s][a][j]*(self.vs[s][j]-self.m[a]) for j in range(3)),I(0)) for a in range(3)] for s in range(2)]
   self.data.append((Q,ell,g))
 def coeff(self,k,b,u,kind='v'):
  if kind=='v':
   return [((u*a3+a2)*u+a1)*u+a0 for a0,a1,a2,a3 in self.cp[k][b]]
  return [((u*(3*a3)+2*a2)*u+a1)/self.dt for a0,a1,a2,a3 in self.cp[k][b]]
 def residual(self,k,p,u):
  co=[self.coeff(k,b,u) for b in range(5)]
  V=[cheb(cs,p) for cs in co]
  vt=[cheb(self.coeff(k,b,u,'t'),p) for b in range(5)]
  diff=I('.5')*(self.sigma*p*(1-p))**2
  _,ell,g=self.data[k]
  results=[]
  for b in range(5):
   adv=[I(0)]
   for a in range(1,min(b,2)+1):
    w=p*(ell[1][a]-ell[0][a])+ell[0][a]
    gg=p*(g[1][a]-g[0][a])+g[0][a]
    win=p*ell[1][a]/w;loss=p*(1-ell[1][a])/(1-w)
    adv.append(gg+w*cheb(co[b-a],win)+(1-w)*cheb(co[b],loss)-V[b])
   h=self.eps*iv.log(sum([iv.exp(z/self.eps) for z in adv],I(0)))
   results.append(vt[b]+diff*cheb2(co[b],p)+self.lam*h)
  return results

if __name__=='__main__':
 path=sys.argv[1];cert=Cert(path)
 for width in ('1','0.1','0.01','0.001','0.0001'):
  p=box('.5',str(.5+float(width)))
  u=box('.5',str(.5+float(width)))
  t0=time.time();r=cert.residual(30,p,u)
  print('width',width,'enclosure max',max(norm_ub(x) for x in r),'per budget',[round(norm_ub(x),9) for x in r], 'sec',time.time()-t0,flush=True)
