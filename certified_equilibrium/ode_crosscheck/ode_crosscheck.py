"""Independent floating-point cross-check; NOT rigorous interval verification.
Uses SciPy's DOP853 to solve the 10-state regime-budget ODE with the archived
fixed-flow Chebyshev-Hermite candidate policy. No imports from original v8 code.
"""
import json, sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit,softmax
from numpy.polynomial.chebyshev import chebval
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'..'/'candidates'
B=np.array([0.,.05,.10]);sigma=.2;lam=2.25;eps=.05

def run(tag):
 obj=json.load(open(ROOT/f'candidate_eta0.{tag}_n7.json'))
 eta=float(obj['eta']); q=np.array(obj['q'],float)
 co=np.array(obj['co'],float); d=np.array(obj['d'],float)
 w=np.zeros((2,3,3));vv=np.zeros((2,3,3));
 for s,ds in enumerate((-.01,.01)):
  for a in (1,2):
   for m in range(3):
    w[s,a,m]=expit((B[a]-B[m]-ds)/.2)
    delta=(.060 if s==0 else .095)+.03*B[m]
    vv[s,a,m]=w[s,a,m]*(delta-B[a])
 m=np.zeros(10,float);m[4]=m[9]=.5
 maxerr=0; arg=None; sample_count=0
 for k in range(60):
  Q=(1-eta)*np.array([.2,.4,.4])+eta*np.r_[q[k],1-q[k].sum()]
  ell=np.einsum('sam,m->sa',w,Q)
  g=np.einsum('sam,m->sa',vv,Q)
  def policy(u):
   C=[]; du=1/60
   for b in range(5):
    y0=co[k,b];y1=co[k+1,b];D0=d[k,0,b];D1=d[k,1,b]
    coeff=y0+(du*D0)*u+(3*(y1-y0)-du*(2*D0+D1))*u*u+(2*(y0-y1)+du*(D0+D1))*u**3
    C.append(coeff)
   p=.5;V=np.array([chebval(2*p-1,C[b]) for b in range(5)])
   ans=np.zeros((5,3));
   for b in range(5):
    score=[0.]
    for a in range(1,min(b,2)+1):
     winprob=(1-p)*ell[0,a]+p*ell[1,a]
     pw=p*ell[1,a]/winprob; pl=p*(1-ell[1,a])/(1-winprob)
     cont=winprob*chebval(2*pw-1,C[b-a])+(1-winprob)*chebval(2*pl-1,C[b])
     rew=(1-p)*g[0,a]+p*g[1,a]
     score.append(rew+cont-V[b])
    ans[b,:len(score)]=softmax(np.array(score)/eps)
   return ans
  def f(t,z):
   u=min(max((t-k/60)*60,0.),1.)
   po=policy(u)
   dz=np.zeros((2,5))
   mass=z.reshape(2,5)
   for s in range(2):
    for b in range(5):
     for a in range(1,min(b,2)+1):
      rate=lam*po[b,a]*ell[s,a]
      dz[s,b]-=mass[s,b]*rate
      dz[s,b-a]+=mass[s,b]*rate
   return dz.ravel()
  ts=np.linspace(k/60,(k+1)/60,101)
  sol=solve_ivp(f,(k/60,(k+1)/60),m,method='DOP853',t_eval=ts,rtol=1e-12,atol=1e-14)
  if not sol.success:raise RuntimeError(sol.message)
  for n in range(100):
   t=ts[n]; mass=sol.y[:,n].reshape(2,5);po=policy(n/100)
   qref=(mass.sum(axis=0)[:,None]*po).sum(axis=0)
   qcand=np.r_[q[k],1-q[k].sum()]
   dist=np.abs(qref-qcand).sum()/(1+t)
   if dist>maxerr:maxerr=float(dist);arg=[k,n,float(t)]
   sample_count+=1
  m=sol.y[:,-1]
 return {'eta':eta,'n_nodes':sample_count,'max_node_error':maxerr,'argmax':arg,'final_total_mass':float(m.sum()),'final_mass_by_regime_budget':m.reshape(2,5).tolist()}
if __name__=='__main__':
 rows=[run(tag) for tag in ('50','60')]
 print(json.dumps(rows,indent=2));open(str(Path(__file__).resolve().parent/'ode_crosscheck_results.json'),'w').write(json.dumps(rows,indent=2))
