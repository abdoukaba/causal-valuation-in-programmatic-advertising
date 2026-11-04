from iv_taylor import C2,D2,O,box,norm_ub
from decimal import Decimal as D
import sys,math

def pol_jets(c,k,p,u):
 co=[c.coeff(k,b,u) for b in range(5)]
 from iv_taylor import chebJ
 V=[chebJ(cs,p) for cs in co]
 _,ell,g=c.data[k]
 jets=[]
 for b in range(1,5):
  adv=[D2(0)]
  for a in range(1,min(b,2)+1):
   w=p*(ell[1][a]-ell[0][a])+ell[0][a]
   gg=p*(g[1][a]-g[0][a])+g[0][a]
   win=p*ell[1][a]/w;loss=p*(1-ell[1][a])/(1-w)
   adv.append(gg+w*chebJ(co[b-a],win)+(1-w)*chebJ(co[b],loss)-V[b])
  e=[D2.exp(z/c.eps) for z in adv]
  z=sum(e,D2(0)); prob=[v/z for v in e]
  jets.extend(prob)
 return jets

def lip_bound(c,k,pc,uc,rp,ru):
 p0=D2(str(pc),p=O);u0=D2(str(uc),u=O)
 p=D2(box(pc-rp,pc+rp),p=O);u=D2(box(uc-ru,uc+ru),u=O)
 cen=pol_jets(c,k,p0,u0);outer=pol_jets(c,k,p,u)
 from decimal import Decimal as D,ROUND_CEILING,localcontext,Context
 import math
 def U(x):return D(str(math.nextafter(norm_ub(x),math.inf)))
 with localcontext(Context(prec=40,rounding=ROUND_CEILING)):
  v=sum((U(x.p)+D(str(rp))*U(y.pp)+D(str(ru))*U(y.pu) for x,y in zip(cen,outer)),D(0))
  # group components by budget. b1 has 2 actions, b2,b3,b4 each 3
  k=0;out=[]
  for count in (2,3,3,3):
   out.append(sum((U(x.p)+D(str(rp))*U(y.pp)+D(str(ru))*U(y.pu) for x,y in zip(cen[k:k+count],outer[k:k+count])),D(0)))
   k+=count
 return max(out),out

def pol_lip(c,k,p,u,full=False):
 jets=pol_jets(c,k,p,u)
 # indices b1 has 2, b2..b4 have 3
 j=0;res=[]
 for count in (2,3,3,3):
  res.append(sum(norm_ub(v.p) for v in jets[j:j+count]));j+=count
 return max(res),res

if __name__=='__main__':
 c=C2(sys.argv[1]);res=[]
 for k in (0,15,30,45,59):
  mx=0
  for j in range(10):
   p= D(2*j+1)/D(20);bnd,arr=pol_lip(c,k,D2(box(p-D('.05'),p+D('.05')),p=O),D2(box(D(0),D(1)),u=O))
   mx=max(mx,bnd)
  print(k,mx,flush=True)
