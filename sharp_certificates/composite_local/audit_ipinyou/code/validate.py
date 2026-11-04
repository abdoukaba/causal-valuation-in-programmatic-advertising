"""Arb continuous-domain residual bound. The proof, units and weak-moment closure are in the report."""
import argparse,json,sys
from fractions import Fraction
from pathlib import Path
from spec import primitives
from flint import arb as A,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
def maxb(seq):
    it=iter(seq);r=next(it)
    for a in it:r=r.max(a)
    return r
def minb(seq):return -maxb(-a for a in seq)
def span(v):return maxb(v)-minb(v)
def rowsum(m):return maxb(sum(abs(m[i,j]) for j in range(m.ncols())) for i in range(m.nrows()))
def upper(x):
    # Integer ceiling of an upper endpoint, with an extra unit in the last printed place.
    return (int((x*10**10).upper().ceil().unique_fmpz())+1)/10**10
def gel(rate):return 1/(1+(-rate).exp())
def tvrow(M,q):
    return sum(abs(M[0,j]-q[j]) for j in range(3))/2
def validate(file,precision,parts):
    ctx.prec=precision;w=json.loads(Path(file).read_text());n=w['cells'];S=w['scale'];dt=A(1)/n
    game=w['case'];eta=A(w['eta'])
    grid,q0,zeta,ds,val,p0=primitives(game,A);eps=A('.05');rate=A('2.25')
    state_prior=[1-p0,p0];vs=[[[A(x)/S for x in row] for row in mat] for mat in w['value_endpoints']]
    assert all(x==0 for row in vs[-1] for x in row);assert all(sum(q)==S and min(q)>=0 for q in w['midpoint_flow']+w['nodal_flow'])
    link=[[[gel((grid[a]-grid[m]-ds[s])/zeta) for m in range(4)] for a in range(3)] for s in (0,1)]
    x=arb_mat(1,10);x[0,8],x[0,9]=state_prior
    x0=arb_mat(x);frozen_local=[];frozen_reference=[]
    variance=A(0);mass_error=A(0);refs=[];deltas=[];timefree=[];moment_errors=[];node_errors=[]
    for i in range(n):
        q=[A(v)/S for v in w['midpoint_flow'][i]];Q=[(1-eta)*q0[m]+(eta*q[m] if m<3 else 0) for m in range(4)]
        ell=[[sum(Q[m]*link[s][a][m] for m in range(4)) for a in range(3)] for s in (0,1)]
        g=[[sum(Q[m]*link[s][a][m]*(val[s][m]-grid[a]) for m in range(4)) for a in range(3)] for s in (0,1)]
        growth=A('.04')/16+rate*maxb((ell[1][a]-ell[0][a])**2/(16*minb(v*(1-v) for v in (ell[0][a],ell[1][a]))) for a in (1,2))
        vmax=variance+dt*growth
        H0=arb_mat(10,3);G0=arb_mat(10,10);lip_center=A(0)
        H=arb_mat(10,3);G=arb_mat(10,10);delta=A(0);Lt=A(0);diffuse=A(0);curvature=A(0)
        for b in range(5):
            actions=list(range(min(2,b)+1))
            zs=[[[A(0) if not a else g[s][a]+ell[s][a]*(vs[i+r][b-a][s]-vs[i+r][b][s]) for a in actions] for s in (0,1)] for r in (0,1)]
            ztime=[[(y-u)/dt for y,u in zip(zs[1][s],zs[0][s])] for s in (0,1)]
            time_range=maxb(span(x1) for x1 in ztime);Lt=Lt.max(time_range/(4*eps))
            belief_range=maxb(span([hi-lo for lo,hi in zip(z[0],z[1])]) for z in zs)
            curv=rate*((time_range*dt)**2+(belief_range/parts)**2)/(32*eps)
            for r in (0,1):
                for k in range(parts+1):
                    p=A(k)/parts
                    z=[(1-p)*lo+p*hi for lo,hi in zip(*zs[r])]
                    vt=sum(((1-p) if s==0 else p)*(vs[i+1][b][s]-vs[i][b][s])/dt for s in (0,1))
                    R=vt+rate*eps*sum((a/eps).exp() for a in z).log()
                    delta=delta.max(abs(R)+curv)
            zmid=[[sum(zs[r][s][a] for r in (0,1))/2 for a in actions] for s in (0,1)]
            ex=[(((1-p0)*lo+p0*hi)/eps).exp() for lo,hi in zip(*zmid)];den=sum(ex);pi=[e/den for e in ex]
            sl=[(hi-lo)/eps for lo,hi in zip(*zmid)];mu=sum(h*d for h,d in zip(pi,sl));der=[h*(d-mu) for h,d in zip(pi,sl)]
            lip_center=lip_center.max(span(sl)/4)
            L2=span(sl)**2/2  # bound on sum_a |d^2 pi_a / dp^2|
            curvature=curvature.max(L2)
            diffuse=diffuse.max(2*sum(abs(d) for d in der)+L2/2)
            for s in (0,1):
                row=2*b+s
                for a in actions:
                    H[row,a]=pi[a]+(s-p0)*der[a];H0[row,a]=pi[a]
                for a in actions[1:]:
                    r0=rate*ell[s][a]*pi[a];G0[row,2*(b-a)+s]+=r0;G0[row,row]-=r0
                    for j in (0,1):
                        joint=(pi[a] if j==s else A(0))+state_prior[j]*der[a]*(s-p0)
                        v=rate*ell[j][a]*joint
                        G[row,2*(b-a)+j]+=v;G[row,2*b+j]-=v
        x0end=x0*(G0*dt).exp()
        frozen_reference.append(tvrow(x0*H0,q).max(tvrow(x0end*H0,q))+rate**2*dt**2/4)
        frozen_local.append(lip_center*vmax.min(p0*(1-p0)).sqrt())
        x0=x0end
        lognorm=maxb(G[j,j]+sum(abs(G[j,k]) for k in range(10) if k!=j) for j in range(10)).max(A(0))
        expansion=(lognorm*dt).exp()
        err=(mass_error+dt*2*rate*maxb(ell[s][a] for s in (0,1) for a in (1,2))*diffuse*vmax)*expansion
        hnorm=rowsum(H);oper=rowsum(G)
        reference_start=x*H;x_end=x*(G*dt).exp();reference_end=x_end*H
        interp=rowsum(x)*expansion*oper**2*hnorm*dt**2/16
        ref=tvrow(reference_start,q).max(tvrow(reference_end,q))+interp
        closure=err*hnorm/2+curvature*vmax/4
        refs.append(ref+closure);moment_errors.append(closure);node_errors.append(ref)
        deltas.append(delta);timefree.append(Lt*dt/2);x=x_end;variance=vmax;mass_error=err
    verror=A(0);regret=A(0);deviations=[None]*n
    for i in reversed(range(n)):
        verror+=dt*deltas[i];deviations[i]=verror/(2*eps)+timefree[i]
        regret+=rate*dt*verror**2/(2*eps)
    mismatch=A(0);frozen_mismatch=A(0);res=[]
    for r,e,fc,fr in zip(refs,deviations,frozen_local,frozen_reference):
        mismatch+=rate*dt*e;frozen_mismatch+=rate*dt*(e+fc)
        res.append((r+e+mismatch).min(fr+fc+e+frozen_mismatch).min(A(1)))
    # Exact TV distance from rounded linear display paths to verification midpoint steps.
    # The convex norm attains its maximum at a subinterval endpoint, including both one-sided traces.
    nodes=w['nodal_flow'];nt=len(nodes)-1
    def flow(t):
        u=t*nt;l=min(int(u),nt-1);v=u-l
        return [(1-v)*a+v*b for a,b in zip(nodes[l],nodes[l+1])]
    diff=Fraction(0)
    for i,q in enumerate(w['midpoint_flow']):
        left=Fraction(i,n);right=Fraction(i+1,n)
        ticks=[left,right]+[Fraction(j,nt) for j in range(int(left*nt)+1,int(right*nt)+1)]
        diff=max(diff,max(sum(abs(a-b) for a,b in zip(q,flow(t)))/(2*S) for t in ticks))
    transfer=A(diff.numerator)/diff.denominator
    rec=dict(case=game,method=w['method'],eta=w['eta'],precision=precision,time_cells=n,belief_pieces=parts,
             max_HJB=upper(maxb(deltas)),value_error_at_zero=upper(verror),witness_policy_fixed_flow_regret=upper(regret),policy_coupling=upper(maxb(deviations)+mismatch),
             max_reference=upper(maxb(node_errors)),max_moment_closure=upper(maxb(moment_errors)),step_residual_TV=upper(maxb(res)),
             displayed_flow_transfer=upper(transfer),residual_cells=[upper(y) for y in res])
    if game in ('ipinyou','t9'):
        f=OUT/('gain_'+game+'_'+format(float(eta),'.2f')[1:]+'.json');gain=json.loads(f.read_text())
        low,hi=gain['outer_boxes'][-1]
        assert all(low[a]*S<=row[a]*1000000<=hi[a]*S for row in w['midpoint_flow'] for a in range(3))
        gain_upper=A(gain['unweighted_bound']);assert gain_upper<1
        rec['simple_distance_TV']=upper(maxb(res)/(1-gain_upper)+transfer)
    return rec
if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('file');cli.add_argument('--bits',type=int,default=160);cli.add_argument('--parts',type=int,default=12)
    a=cli.parse_args();out=validate(a.file,a.bits,a.parts)
    path=Path(a.file).with_name(Path(a.file).stem.replace('witness','certificate')+'.json')
    path.write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k!='residual_cells'})
