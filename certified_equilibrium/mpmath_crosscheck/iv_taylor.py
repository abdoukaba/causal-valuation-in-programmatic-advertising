from iv_hjb import Cert,iv,I,box,norm_ub
import sys,time
iv.dps=35
Z=iv.mpf(0);O=iv.mpf(1)
class D2:
 __slots__=('v','p','u','pp','pu','uu')
 def __init__(self,v,p=None,u=None,pp=None,pu=None,uu=None):
  if isinstance(v,D2):
   self.v,self.p,self.u,self.pp,self.pu,self.uu=v.v,v.p,v.u,v.pp,v.pu,v.uu
  else:
   self.v=I(v);self.p=Z if p is None else I(p);self.u=Z if u is None else I(u)
   self.pp=Z if pp is None else I(pp);self.pu=Z if pu is None else I(pu);self.uu=Z if uu is None else I(uu)
 @staticmethod
 def vjp(v):return v if isinstance(v,D2) else D2(v)
 def __add__(a,b):
  b=D2.vjp(b)
  return D2(a.v+b.v,a.p+b.p,a.u+b.u,a.pp+b.pp,a.pu+b.pu,a.uu+b.uu)
 __radd__=__add__
 def __neg__(a):return D2(-a.v,-a.p,-a.u,-a.pp,-a.pu,-a.uu)
 def __sub__(a,b):return a+-D2.vjp(b)
 def __rsub__(a,b):return D2.vjp(b)+-a
 def __mul__(a,b):
  b=D2.vjp(b);A=a;B=b
  return D2(A.v*B.v,A.p*B.v+A.v*B.p,A.u*B.v+A.v*B.u,
    A.pp*B.v+2*A.p*B.p+A.v*B.pp,
    A.pu*B.v+A.p*B.u+A.u*B.p+A.v*B.pu,
    A.uu*B.v+2*A.u*B.u+A.v*B.uu)
 __rmul__=__mul__
 def inverse(a):
  b=O/a.v;b2=b*b;b3=b2*b
  return D2(b,-a.p*b2,-a.u*b2,2*a.p*a.p*b3-a.pp*b2,2*a.p*a.u*b3-a.pu*b2,2*a.u*a.u*b3-a.uu*b2)
 def __truediv__(a,b):return a*D2.vjp(b).inverse()
 def __rtruediv__(a,b):return D2.vjp(b)*a.inverse()
 def __pow__(a,n):
  assert isinstance(n,int) and n>=0
  if not n:return D2(1)
  if n==1:return a
  if n%2:return a*(a**(n-1))
  b=a**(n//2);return b*b
 def exp(a):
  b=iv.exp(a.v)
  return D2(b,b*a.p,b*a.u,b*(a.pp+a.p*a.p),b*(a.pu+a.p*a.u),b*(a.uu+a.u*a.u))
 def log(a):
  b=O/a.v;b2=b*b
  return D2(iv.log(a.v),a.p*b,a.u*b,a.pp*b-a.p*a.p*b2,a.pu*b-a.p*a.u*b2,a.uu*b-a.u*a.u*b2)

def chebJ(cs,p):
 z=2*p-1
 t0=D2(1);t1=z
 if len(cs)==1:return D2.vjp(cs[0])
 out=cs[0]+cs[1]*t1
 for c in cs[2:]:
  t0,t1=t1,2*z*t1-t0
  out+=c*t1
 return out

def chebppJ(cs,p):
 z=2*p-1;a=D2(1);ad=D2(0);add=D2(0);b=z;bd=D2(1);bdd=D2(0);out=D2(0)
 for c in cs[2:]:
  cn=2*z*b-a; cd=2*b+2*z*bd-ad;cdd=4*bd+2*z*bdd-add
  out+=c*cdd; a,ad,add=b,bd,bdd;b,bd,bdd=cn,cd,cdd
 return 4*out

class C2(Cert):
 def residualJ(self,k,p,u):
  coeff=[self.coeff(k,b,u) for b in range(5)]
  V=[chebJ(cs,p) for cs in coeff]
  vt=[chebJ(self.coeff(k,b,u,'t'),p) for b in range(5)]
  diff=D2('.5')*(D2(self.sigma)*p*(1-p))**2
  _,ell,g=self.data[k];res=[]
  for b in range(5):
   adv=[D2(0)]
   for a in range(1,min(b,2)+1):
    w=p*(ell[1][a]-ell[0][a])+ell[0][a]
    gg=p*(g[1][a]-g[0][a])+g[0][a]
    win=p*ell[1][a]/w;loss=p*(1-ell[1][a])/(1-w)
    adv.append(gg+w*chebJ(coeff[b-a],win)+(1-w)*chebJ(coeff[b],loss)-V[b])
   h=(D2(self.eps)*sum([D2.exp(z/self.eps) for z in adv],D2(0)).log())
   res.append(vt[b]+diff*chebppJ(coeff[b],p)+D2(self.lam)*h)
  return res

def certbound(c,k,pc,uc,rp,ru):
 mid=c.residualJ(k,D2(str(pc),p=O),D2(str(uc),u=O))
 big=c.residualJ(k,D2(box(pc-rp,pc+rp),p=O),D2(box(uc-ru,uc+ru),u=O))
 R=max(norm_ub(x.v)+rp*norm_ub(x.p)+ru*norm_ub(x.u)+rp*rp/2*norm_ub(y.pp)+rp*ru*norm_ub(y.pu)+ru*ru/2*norm_ub(y.uu) for x,y in zip(mid,big))
 return R,[(norm_ub(y.pp),norm_ub(y.pu),norm_ub(y.uu)) for y in big]

if __name__=='__main__':
 c=C2(sys.argv[1]);k=int(sys.argv[2]) if len(sys.argv)>2 else 30
 for rp,ru in ((.5,.5),(.05,.5),(.02,.1),(.01,.05),(.005,.05),(.005,.025)):
  t=time.time();R,arr=certbound(c,k,.5,.5,rp,ru)
  print(rp,ru,'bound',R,'hessmax',max(y[0] for y in arr),max(y[1] for y in arr),max(y[2] for y in arr),'elapsed',time.time()-t,flush=True)
