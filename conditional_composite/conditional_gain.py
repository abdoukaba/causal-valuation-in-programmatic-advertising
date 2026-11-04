"""Exploratory exact-model comparison using conditional continuation payoffs.
Float mode is search only. No claimed certificate from this file alone.
"""
import sys,json,itertools,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'v5/code/sharp_certificates/comparison/proof'))
import budget_gain as bg

def polyverts(lo,hi):
    n=len(lo)
    if n==1:return [np.ones(1)]
    out=[]
    for free in range(n):
        idx=[j for j in range(n) if j!=free]
        for bits in itertools.product([0,1],repeat=n-1):
            x=np.zeros(n)
            for j,v in zip(idx,bits):x[j]=hi[j] if v else lo[j]
            x[free]=1-sum(x)
            if lo[free]-1e-12<=x[free]<=hi[free]+1e-12:out.append(x)
    if not out:raise ValueError((lo,hi))
    return out

def stage_min(scores,lo,hi,eps):
    return min(x@scores-eps*sum(v*np.log(v) for v in x if v>0) for x in polyverts(lo,hi))

def setup(box,eta=.45):
    grid,q,z,d,D=bg.stored('archived')
    grid,q,d,D=np.array(grid,float),np.array(q,float),np.array(d,float),np.array(D,float)
    w=np.zeros((2,3,4))
    for s in range(2):
        for a in [1,2]:w[s,a]=1/(1+np.exp(-(grid[a]-grid-d[s])/float(z)))
    g=w*(D[:,None,:]-grid[None,:3,None])
    vs=np.array(bg.vertices(*box))/1e6
    Q=(1-eta)*q[None,:]+eta*np.pad(vs,((0,0),(0,1)))
    ell=np.einsum('vk,sak->vsa',Q,w);rew=np.einsum('vk,sak->vsa',Q,g)
    return w,g,ell,rew

def improve(state,box,rounds=8):
    n=len(state.c);eps=float(state.e);lam=float(state.la);dt=1/n
    no=np.exp(-lam*dt);hit=-np.expm1(-lam*dt)
    w,g,ell,rew=setup(box)
    nv=len(ell)
    # Original bounds hold uniformly over q and belief, and stay valid in refinements.
    pls=[[np.array(r.policy_lower,float) for r in c.row] for c in state.c]
    pus=[[np.array(r.policy_upper,float) for r in c.row] for c in state.c]
    cells=None
    for iteration in range(rounds):
        rightL=np.zeros((2,5));rightU=np.zeros((2,5));cells=[None]*n
        for k in reversed(range(n)):
            cl=np.zeros((2,5));cu=np.zeros((2,5))
            for s in range(2):
                for b in range(1,5):
                    na=min(b,2)+1
                    # Conditional stage bound, plus event/no-event budget hull.
                    rlo=min(stage_min(rew[v,s,:min(bp,2)+1],pls[k][bp],pus[k][bp],eps) for v in range(nv) for bp in range(b+1))
                    rhi=max(eps*logsumexp(rew[v,s,:na]/eps) for v in range(nv))
                    cl[s,b]=no*rightL[s,b]+hit*min(rightL[s,:b+1])+lam*dt*min(0,rlo)
                    cu[s,b]=no*rightU[s,b]+hit*max(rightU[s,:b+1])+lam*dt*max(0,rhi)
            # cell bounds on conditional action scores, all posterior priors covered
            tl=np.zeros((nv,2,5,3));tu=tl.copy()
            for b in range(5):
                for a in range(min(b,2)+1):
                    tl[:,:,b,a]=rew[:,:,a]+(1-ell[:,:,a])*cl[:,b]+ell[:,:,a]*cl[:,b-a]
                    tu[:,:,b,a]=rew[:,:,a]+(1-ell[:,:,a])*cu[:,b]+ell[:,:,a]*cu[:,b-a]
                na=min(b,2)+1
                for a in range(na):
                    # Difference bounds at a common s, common prior, common Q.
                    dl=np.array([np.min(tl[:,:,b,j]-tu[:,:,b,a]) for j in range(na)])
                    du=np.array([np.max(tu[:,:,b,j]-tl[:,:,b,a]) for j in range(na)])
                    dl[a]=du[a]=0
                    pls[k][b][a]=max(pls[k][b][a],1/sum(np.exp(du/eps)))
                    pus[k][b][a]=min(pus[k][b][a],1/sum(np.exp(dl/eps)))
            leftL=np.zeros((2,5));leftU=np.zeros((2,5))
            for s in range(2):
                for b in range(1,5):
                    na=min(b,2)+1
                    leftL[s,b]=no*rightL[s,b]+hit*min(stage_min(tl[v,s,b,:na],pls[k][b],pus[k][b],eps) for v in range(nv))
                    leftU[s,b]=no*rightU[s,b]+hit*max(eps*logsumexp(tu[v,s,b,:na]/eps) for v in range(nv))
            cells[k]=(cl,cu,tl,tu)
            rightL,rightU=leftL,leftU
        # print is only diagnostic
    for k,c in enumerate(state.c):
        cl,cu,tl,tu=cells[k]
        for b,row in enumerate(c.row):
            na=min(b,2)+1
            row.policy_lower=list(pls[k][b]);row.policy_upper=list(pus[k][b])
            row.gibbs_factor=[min(.25,x*(1-x) if x>.5 else .25,y*(1-y) if y<.5 else .25) for x,y in zip(pls[k][b],pus[k][b])]
            for a in range(na):
                if na==1:row.policy_lipschitz[a]=0;continue
                # S'_a-S'_j=(T1a-T1j)-(T0a-T0j), via perspective envelopes.
                pair=[]
                for j in range(na):
                    if j==a:continue
                    upper=tu[:,1,b,a]-tl[:,1,b,j]-tl[:,0,b,a]+tu[:,0,b,j]
                    lower=tl[:,1,b,a]-tu[:,1,b,j]-tu[:,0,b,a]+tl[:,0,b,j]
                    pair.append(max(np.max(upper),-np.min(lower)))
                row.policy_lipschitz[a]=min(row.policy_lipschitz[a],row.gibbs_factor[a]/eps*max(pair))
            forces=[]
            for v,(i,j) in enumerate(bg.edges):
                fl=np.zeros((2,na));fu=fl.copy()
                for a in range(1,na):
                    D=w[:,a,i]-w[:,a,j];E=g[:,a,i]-g[:,a,j]
                    fl[:,a]=.45*(E+D*(cl[:,b-a]-cu[:,b]))
                    fu[:,a]=.45*(E+D*(cu[:,b-a]-cl[:,b]))
                forces.append((fl,fu))
                lows=np.min(fl,axis=0);highs=np.max(fu,axis=0)
                avlo=state.math_ops.box_min(lows,pls[k][b],pus[k][b])
                avhi=state.math_ops.box_max(highs,pls[k][b],pus[k][b])
                oldlo,oldhi=row.average_forcing[v]
                row.average_forcing[v]=(max(oldlo,avlo),min(oldhi,avhi))
            for a in range(na):
                if na==1:continue
                sep=max(max(np.max(fu[:,a]-fl[:,j]),np.max(fu[:,j]-fl[:,a])) for fl,fu in forces for j in range(na) if a!=j)
                row.instant_gain[a]=min(row.instant_gain[a],row.gibbs_factor[a]/eps*sep)
        c.forcing_span=[max(max(c.row[i].average_forcing[v][1] for i in range(b+1))-min(c.row[i].average_forcing[v][0] for i in range(b+1)) for v in range(3)) for b in range(5)]
        c.spend_one_or_more=state.math_ops.maximum(state.math_ops.box_max([0]+[state.a[a].hi for a in range(1,len(r.policy_lower))],r.policy_lower,r.policy_upper) for r in c.row)
        c.spend_two=max([0]+[r.policy_upper[2]*state.a[2].hi for r in c.row if len(r.policy_lower)==3])
    mu1=mu2=0
    for c in state.c:
        mu1+=lam*dt*(c.spend_one_or_more-c.spend_two);mu2+=lam*dt*c.spend_two
        h0=np.exp(-mu1-mu2)
        ps=[h0,h0*mu1,h0*(mu1**2/2+mu2),h0*(mu1**3/6+mu1*mu2)]
        c.probs=ps+[1-sum(ps)]
    return state,{'conditional_lower_t0':rightL.tolist(),'conditional_upper_t0':rightU.tolist(),
        'policy_lower_t0':[x.tolist() for x in pls[0]],'policy_upper_t0':[x.tolist() for x in pus[0]],
        'instant_t0':[r.instant_gain for r in state.c[0].row],
        'policy_lip_t0':[r.policy_lipschitz for r in state.c[0].row]}

def run(n=40,rounds=8,tube=None):
    box=([0,0,0],[1000000]*3)
    for _ in range(2):
        st=bg.model('archived','.45',n,False,box);lo,hi=bg.restrict(st)
        box=([max(x,y) for x,y in zip(box[0],lo)],[min(x,y) for x,y in zip(box[1],hi)])
    if tube is not None:
        flow=json.loads((ROOT/'v5/code/sharp_certificates/composite_local/data/fem_path.json').read_text())
        arr=np.array(flow['nodal_flow'])/flow['scale']
        box=([max(0,int(np.floor((arr[:,j].min()-tube)*1e6))) for j in range(3)],
             [min(1000000,int(np.ceil((arr[:,j].max()+tube)*1e6))) for j in range(3)])
    st=bg.model('archived','.45',n,False,box)
    st,diag=improve(st,box,rounds)
    K=bg.kernels(st);weight=np.ones(n)
    for _ in range(45):
        v=np.array(bg.evalw(st,K,weight));weight=v/v.max()
    v=bg.evalw(st,K,weight)
    result=dict(cells=n,rounds=rounds,tube=tube,box=box,bound=max(v/weight),unweighted=max(bg.evalw(st,K,np.ones(n))),diagnostics=diag,scope='exploratory floating point; mathematical inequalities require audit and interval implementation')
    return result
if __name__=='__main__':
    t=time.time();r=run(int(sys.argv[1]) if len(sys.argv)>1 else 40,int(sys.argv[2]) if len(sys.argv)>2 else 8,float(sys.argv[3]) if len(sys.argv)>3 else None)
    print(json.dumps(r,indent=2));print('seconds',time.time()-t,file=sys.stderr)
