"""Continuous-belief fixed-flow verification by affine dual test functions.
The fit only proposes integer coefficients. Validation uses exclusively
these integers, archived rational inputs and Arb enclosures on full cells.
"""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import numpy as np
from flint import arb as A, arb_mat as M, ctx
from verify_grid import upper_decimal,maxupper
ROOT=Path(__file__).resolve().parent
WITNESS_SCALE=10**12

def stages():
    src=json.loads((ROOT/'data/witness_fem_composite_0.45.json').read_text())
    inp=json.loads((ROOT/'data/archived.json').read_text(),parse_float=str)['criteo_offsets']['primitives']
    S,N=src['scale'],src['cells']
    assert N==len(src['midpoint_flow'])==len(src['value_endpoints'])-1
    eta=A('.45');base=list(map(A,inp['q0']))
    vals=[[A(v) for v in row] for row in inp['Delta']]
    link=[[[1/(1+((A(m-a)/20+A(inp['d'][s]))/A(inp['zeta'])).exp()) for m in range(4)] for a in range(3)] for s in (0,1)]
    result=[]
    for i in range(N):
        q=[A(k)/S for k in src['midpoint_flow'][i]]
        mix=[(1-eta)*base[m]+(eta*q[m] if m<3 else 0) for m in range(4)]
        wins=[[sum(w*v for w,v in zip(link[s][a],mix)) for s in (0,1)] for a in (0,1,2)]
        v=[[A(src['value_endpoints'][i][b][s]+src['value_endpoints'][i+1][b][s])/(2*S) for s in (0,1)] for b in range(5)]
        score=[]
        for b in range(5):
            rows=[[A(0),A(0)]]
            for a in range(1,min(2,b)+1):
                rows.append([20*(sum(mix[k]*link[s][a][k]*(vals[s][k]-A(a)/20) for k in range(4))+
                              wins[a][s]*(v[b-a][s]-v[b][s])) for s in (0,1)])
            score.append(rows)
        result.append((wins,score,q))
    return result,A(inp['p0'])

def propose(U,groups):
    # Pointwise least-squares slope with a midrange-adjusted affine intercept.
    p=np.linspace(0,1,65);mid=p-.5
    def fit(y):
        slope=np.einsum('p,bpj->bj',mid,y)/np.dot(mid,mid)
        rem=y-p[None,:,None]*slope[:,None,:]
        low=(rem.min(axis=1)+rem.max(axis=1))/2
        return np.stack([low,low+slope],axis=1)
    controls=[]
    for k in range(groups):
        wins,sc,_=U[(2*k+1)*len(U)//(2*groups)]
        w=np.array(wins,float)
        pi=np.zeros((5,len(p),3))
        for b,row in enumerate(sc):
            z=np.array(row,float)
            ex=np.exp(z[:,0]+p[:,None]*(z[:,1]-z[:,0]))
            pi[b,:,:len(row)]=ex/ex.sum(axis=1)[:,None]
        controls.append((w,pi))
    def drift(v,k):
        w,pi=controls[k];res=np.zeros((5,len(p),3))
        for b in range(1,5):
            for a in range(1,min(2,b)+1):
                diff=v[b-a]-v[b]
                stage=(1-p[:,None])*w[a,0]*diff[0]+p[:,None]*w[a,1]*diff[1]
                res[b]+=2.25*pi[b,:,a,None]*stage
        return fit(res)
    blocks=[]
    for k,(_,pi) in enumerate(controls):
        v=fit(pi);vs=[v]
        for l in reversed(range(k)):
            pred=v+drift(v,l)/(2*groups)
            v=v+drift(pred,l)/groups
            vs.append(v)
        blocks.append(np.rint(np.array(vs[::-1])*WITNESS_SCALE).astype('int64').tolist())
    return dict(groups=groups,scale=WITNESS_SCALE,blocks=blocks)

def hull(seq):
    it=iter(seq);z=next(it)
    for x in it:z=z.union(x)
    return z

def cell_tables(U,groups,parts):
    grids=[(A(i)/parts).union(A(i+1)/parts) for i in range(parts)]
    group_size=len(U)//groups
    for k in range(groups):
        group=U[k*group_size:(k+1)*group_size];h=[];mat=[]
        wr=[[hull(x[0][a][s] for x in group) for s in (0,1)] for a in range(3)]
        q=[hull(x[2][j] for x in group) for j in range(3)]
        for b in range(5):
            # Scores at end beliefs. Enclose affine score DIFFERENCES before extrapolating.
            num=min(2,b)+1
            dif=[[([hull(x[1][b][j][s]-x[1][b][a][s] for x in group) for s in (0,1)],
                    hull((x[1][b][j][1]-x[1][b][a][1])-
                         (x[1][b][j][0]-x[1][b][a][0]) for x in group))
                  for j in range(num)] for a in range(num)]
            terms=M(parts,6);pol=M(parts,3)
            for i,p in enumerate(grids):
                terms[i,0]=1;terms[i,1]=p
                for a in range(num):
                    total=A(1)
                    for j in range(num):
                        if a==j:continue
                        ends,slope=dif[a][j]
                        z=(ends[0]+p*slope).intersection(ends[1]-(1-p)*slope)
                        total+=z.exp()
                    pol[i,a]=1/total
                    if a:
                        terms[i,2*a]=A('2.25')*pol[i,a]*(1-p)*wr[a][0]
                        terms[i,2*a+1]=A('2.25')*pol[i,a]*p*wr[a][1]
            h.append(pol);mat.append(terms)
        yield mat,h,q,grids

def check(frozen,U,p0,parts):
    G=frozen['groups'];scale=frozen['scale']
    assert len(U)%G==0 and len(frozen['blocks'])==G
    tables=list(cell_tables(U,G,parts))
    coupling=json.loads((ROOT/'audit_ipinyou/results/certificate_fem_composite_0.45.json').read_text(),parse_float=str)
    assert coupling['method']=='fem' and coupling['case']=='composite' and coupling['time_cells']==len(U)
    polerr=A(coupling['policy_coupling'])
    reports=[]
    for k,block in enumerate(frozen['blocks']):
        assert len(block)==k+1
        vals=[[[[A(x)/scale for x in r] for r in budget] for budget in step] for step in block]
        dist=[A(0)]*3;max_terminal=[A(0)]*3
        for l in range(k+1):
            mats,policy,q,grids=tables[l]
            left=vals[l];right=vals[min(l+1,k)]
            errors=[A(0)]*3
            for b in range(5):
                dt=[[G*(right[b][s][j]-left[b][s][j]) for j in range(3)] for s in (0,1)]
                # Affine in test time, frozen policy enclosures hold across the entire cell.
                for v in ((left,right) if l<k else (left,)):
                    coeff=M(6,3)
                    for j in range(3):
                        coeff[0,j]=dt[0][j];coeff[1,j]=dt[1][j]-dt[0][j]
                        for a in range(1,min(2,b)+1):
                            for s in (0,1):
                                coeff[2*a+s,j]=v[b-a][s][j]-v[b][s][j]
                    resid=mats[b]*coeff
                    for j in range(3):
                        errors[j]=errors[j].max(maxupper(abs(resid[i,j]) for i in range(parts)))
                if l==k:
                    for j in range(3):
                        lo,hi=left[b][0][j],left[b][1][j]
                        max_terminal[j]=max_terminal[j].max(maxupper(
                            abs(policy[b][i,j]-(lo+p*(hi-lo))) for i,p in enumerate(grids)))
            dist=[v+w/G for v,w in zip(dist,errors)]
        start=vals[0][4]
        prior=[start[0][j]+p0*(start[1][j]-start[0][j]) for j in range(3)]
        reference=[abs(a-b) for a,b in zip(prior,q)]
        frozen_gap=maxupper(e+t+r for e,t,r in zip(dist,max_terminal,reference))
        reports.append(dict(group=k,dynamics=upper_decimal(maxupper(dist)),
            terminal=upper_decimal(maxupper(max_terminal)),reference=upper_decimal(maxupper(reference)),
            surrogate_flow_residual_upper=upper_decimal(frozen_gap),
            exact_response_residual_upper=upper_decimal(frozen_gap+polerr)))
    result=dict(groups=G,parts=parts,bits=ctx.prec,
        scope='fixed FEM midpoint rival flow, full continuous time and belief, max TV over all times',
        policy_coupling_upper=coupling['policy_coupling'],
        exact_response_residual_TV_upper=str(max(Decimal(r['exact_response_residual_upper']) for r in reports)),
        frozen_policy_residual_TV_upper=str(max(Decimal(r['surrogate_flow_residual_upper']) for r in reports)),time_bounds=reports)
    return result

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--groups',type=int,default=40);cli.add_argument('--parts',type=int,default=32)
    cli.add_argument('--bits',type=int,default=160);cli.add_argument('--propose',action='store_true')
    args=cli.parse_args();ctx.prec=args.bits
    U,p0=stages();assert len(U)%args.groups==0
    file=ROOT/f'data/dual_witness_{args.groups}.json'
    if args.propose:file.write_text(json.dumps(propose(U,args.groups),separators=(',',':'))+'\n')
    r=check(json.loads(file.read_text()),U,p0,args.parts)
    (ROOT/f'results/continuous_dual_{args.groups}_{args.parts}_p{args.bits}.json').write_text(json.dumps(r,indent=2)+'\n')
    print({k:v for k,v in r.items() if k!='time_bounds'})
