"""Re-solve the illustrative composite on two grids. Run from results/.
All monetary evaluations exclude entropy and use incremental values.
"""
import sys, json
from pathlib import Path
import numpy as np
from calib_full import primitives, model
from calib_model import CalibModel

class RevealedModel(CalibModel):
    """Campaigns know S at entry. The population has the original type shares."""
    def forward(self,q,policy):
        self._init=(0,0.)
        low_mass,low_q=self._forward(q,policy)
        self._init=(self.np-2,1.)
        high_mass,high_q=self._forward(q,policy)
        return ((1-self.p0)*low_mass+self.p0*high_mass,(1-self.p0)*low_q+self.p0*high_q)
    def value_at_prior(self,v):
        return float((1-self.p0)*v[0,-1,0]+self.p0*v[0,-1,-1])

def construct(case,n,nt):
    p=primitives()
    eps={'eps03':.03,'eps08':.08}.get(case,.05)
    rho=[.05,.9,.05] if case=='anchored' else None
    m=model(p,eps=eps,nb=n,nt=nt,rho=rho)
    if case=='reveal':
        m=RevealedModel(n,nt,.45,eps,1.,2.25,m.mgrid,p['q0'],p['zeta'],p['d'],p['Delta'],p['p0'])
    if case=='pooled':
        pooled=(1-m.p0)*m.w[0]+m.p0*m.w[1]
        m.w[:]=pooled
        _,_,w,l=m.population_market(np.ones(3)/3)
        assert np.max(abs(w[:,1:]-m.p[:,None]))<1e-12 and np.max(abs(l[:,1:]-m.p[:,None]))<1e-12
    if case=='tail4h':
        # Move the censored mass only. Keep its capped values and the fitted offsets.
        m.mgrid=m.mgrid.copy();m.mgrid[-1]=.20
        from scipy.special import expit
        for s in (0,1):
            for j in (1,2):m.w[s,j,-1]=expit((m.actions[j]-.20-m.delta[s])/m.noise)
    if case=='attributed':
        m.value=m.value+.02
    return m

def solve(case,n,nt):
    m=construct(case,n,nt);sol=m.solve(2,tol=2e-10,damping=.55,max_iter=350)
    assert sol['res']<2e-10 and abs(sol['m'].sum((1,2))-1).max()<1e-9 and sol['m'].min()>=-1e-12
    q,pol=sol['q'],sol['pol']
    if case=='attributed':m.value=m.value-.02
    ans=dict(bids=q.mean(0).tolist(),monetary=m.monetary(q,pol),monetary_br=m.monetary(q),residual=sol['res'])
    if case in ('baseline','anchored'):
        for target in (1,2):
            rule=np.zeros_like(pol)
            for b in range(m.nb):rule[:,b,:,min(b,target)]=1
            ans['fixed'+str(target)]=m.monetary(q,rule)
        other=construct('anchored' if case=='baseline' else 'baseline',n,nt)
        ans['alternative']=other.monetary(q,other.backward(q)[1])
    if case=='baseline':
        rev=construct('reveal',n,nt)
        ans['disclosed_against_baseline_best']=rev.monetary(q)
        ans['disclosed_against_baseline_logit']=rev.monetary(q,rev.backward(q)[1])
        ans['disclosed_against_baseline_objective']=rev.value_at_prior(rev.backward(q)[0])
        ans['baseline_objective']=m.value_at_prior(sol['V'])
    return ans
if __name__=='__main__':
    n,nt=map(int,sys.argv[1:3])
    out={case:solve(case,n,nt) for case in ('baseline','reveal','pooled','attributed','eps03','eps08','tail4h','anchored')}
    Path(f'counterfactuals_{n}_{nt}.json').write_text(json.dumps(out,indent=2)+'\n')
