"""Arb verification of a conditional-plan comparison for the exact auction MFG.

No belief grid or candidate equilibrium enters the verification. Decimal inputs
are interpreted as exact rationals; elementary functions use Arb balls. Existing
budget_gain.py supplies the continuous Duhamel aggregation. See the proof note.
"""
import sys,json,time,itertools,hashlib,argparse,gzip,os,platform
from pathlib import Path
from functools import lru_cache
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'v5/code/sharp_certificates/comparison/proof'))
import budget_gain as bg
from flint import arb,ctx
SCALE=10**12
from fractions import Fraction
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    sources: str
    shares: tuple
    policy: str

def metadata(cfg):
    return dict(primitives=cfg.sources, eta_endpoints=cfg.shares, policy=cfg.policy,
                epsilon=os.environ.get('EPS', '.05'))

def log_reference(b, eps, cfg):
    na = min(b, 2) + 1
    if cfg.policy == 'shannon':
        return [arb(0)] * na
    masses = [1, 18, 1][:na]
    return [eps * (arb(x)/sum(masses)).log() for x in masses]


def rounded(x,up=False):
    y=(x*SCALE).upper().ceil() if up else (x*SCALE).lower().floor()
    return int(y.unique_fmpz())

def rational_box(lo,hi):
    l=tuple(max(0,rounded(x)) for x in lo)
    u=tuple(min(SCALE,rounded(x,True)) for x in hi)
    assert sum(l)<=SCALE<=sum(u)
    assert all(a<=b for a,b in zip(l,u))
    return l,u

@lru_cache(maxsize=100000)
def polyterms(lo,hi):
    n=len(lo);vertices=set()
    for free in range(n):
        idx=[j for j in range(n) if j!=free]
        for bits in itertools.product([0,1],repeat=n-1):
            x=[0]*n
            for j,v in zip(idx,bits):x[j]=hi[j] if v else lo[j]
            x[free]=SCALE-sum(x)
            if lo[free]<=x[free]<=hi[free]:vertices.add(tuple(x))
    assert vertices
    out=[]
    for row in sorted(vertices):
        x=[arb(v)/SCALE for v in row]
        entropy=-sum((v*v.log() for v,i in zip(x,row) if i),arb(0))
        out.append((x,entropy))
    return out

def stage_min(scores,box,eps,M):
    return M.minimum(sum((x*v for x,v in zip(xs,scores)),arb(0))+eps*h for xs,h in polyterms(*box))

def setup(box,cfg):
    grid,q,z,d,D=bg.stored(cfg.sources)
    w=[[[arb(0) for _ in grid] for _ in range(3)] for _ in range(2)]
    g=[[[arb(0) for _ in grid] for _ in range(3)] for _ in range(2)]
    for s in range(2):
        for a in [1,2]:
            for j,m in enumerate(grid):
                w[s][a][j]=bg.allocation(cfg.sources,grid[a],m,z,d,s,lambda x:x.exp(),arb)
                g[s][a][j]=w[s][a][j]*(D[s][j]-grid[a])
    qs=bg.rival_vertices(q,box,[arb(x) for x in cfg.shares])
    ell=[[[sum((Q[j]*w[s][a][j] for j in range(4)),arb(0)) for a in range(3)] for s in range(2)] for Q in qs]
    rew=[[[sum((Q[j]*g[s][a][j] for j in range(4)),arb(0)) for a in range(3)] for s in range(2)] for Q in qs]
    return w,g,ell,rew

def improve(state,box,cfg,rounds=8):
    M=state.math_ops;z=arb(0);one=arb(1);eps=state.e;lam=state.la;dt=state.h
    n=len(state.c);no=state.no;hit=state.hit
    w,g,ell,rew=setup(box,cfg);nv=len(ell)
    pls=[[list(r.policy_lower) for r in c.row] for c in state.c]
    pus=[[list(r.policy_upper) for r in c.row] for c in state.c]
    offsets=[log_reference(b,eps,cfg) for b in range(5)]
    anchored=cfg.policy!='shannon'
    if anchored:
        # The inherited Shannon bounds do not apply. Initialize from the full
        # feasible simplex and replace all three derivative bounds below.
        for k,c in enumerate(state.c):
            for b,row in enumerate(c.row):
                na=min(b,2)+1
                pls[k][b]=[one if na==1 else z]*na
                pus[k][b]=[one]*na
    def zeros():return [[arb(0) for _ in range(5)] for _ in range(2)]
    for iteration in range(rounds):
        rightL=zeros();rightU=zeros();cells=[None]*n
        for k in reversed(range(n)):
            cl=zeros();cu=zeros()
            pboxes=[rational_box(pls[k][b],pus[k][b]) for b in range(5)]
            for s in range(2):
                rmin=z
                running_upper=z
                for b in range(1,5):
                    na=min(b,2)+1
                    rmin=M.minimum([rmin]+[stage_min([x+y for x,y in zip(rew[v][s][:na],offsets[b])],pboxes[b],eps,M) for v in range(nv)])
                    rhi=M.maximum(eps*sum(((x/eps).exp() for x in [u+y for u,y in zip(rew[v][s][:na],offsets[b])]),z).log() for v in range(nv))
                    running_upper=M.maximum([running_upper,rhi])
                    cl[s][b]=no*rightL[s][b]+hit*M.minimum(rightL[s][:b+1])+lam*dt*rmin
                    cu[s][b]=no*rightU[s][b]+hit*M.maximum(rightU[s][:b+1])+lam*dt*running_upper
            tl=[[[[arb(0) for _ in range(3)] for _ in range(5)] for _ in range(2)] for _ in range(nv)]
            tu=[[[[arb(0) for _ in range(3)] for _ in range(5)] for _ in range(2)] for _ in range(nv)]
            for b in range(5):
                na=min(b,2)+1
                for a in range(na):
                    for v in range(nv):
                        for s in range(2):
                            l=ell[v][s][a];r=rew[v][s][a]+offsets[b][a]
                            tl[v][s][b][a]=r+(1-l)*cl[s][b]+l*cl[s][b-a]
                            tu[v][s][b][a]=r+(1-l)*cu[s][b]+l*cu[s][b-a]
                for a in range(na):
                    dl=[z if j==a else M.minimum(tl[v][s][b][j]-tu[v][s][b][a] for v in range(nv) for s in range(2)) for j in range(na)]
                    du=[z if j==a else M.maximum(tu[v][s][b][j]-tl[v][s][b][a] for v in range(nv) for s in range(2)) for j in range(na)]
                    pls[k][b][a]=M.maximum([pls[k][b][a],one/sum(((x/eps).exp() for x in du),z)])
                    pus[k][b][a]=M.minimum([pus[k][b][a],one/sum(((x/eps).exp() for x in dl),z)])
            pboxes=[rational_box(pls[k][b],pus[k][b]) for b in range(5)]
            leftL=zeros();leftU=zeros()
            for s in range(2):
                for b in range(1,5):
                    na=min(b,2)+1
                    leftL[s][b]=no*rightL[s][b]+hit*M.minimum(stage_min(tl[v][s][b][:na],pboxes[b],eps,M) for v in range(nv))
                    leftU[s][b]=no*rightU[s][b]+hit*M.maximum(eps*sum(((x/eps).exp() for x in tu[v][s][b][:na]),z).log() for v in range(nv))
            cells[k]=(cl,cu,tl,tu)
            rightL,rightU=leftL,leftU
    for k,c in enumerate(state.c):
        cl,cu,tl,tu=cells[k]
        for b,row in enumerate(c.row):
            na=min(b,2)+1
            row.policy_lower=pls[k][b];row.policy_upper=pus[k][b]
            row.gibbs_factor=[M.minimum([one/4,x*(1-x) if x>one/2 else one/4,y*(1-y) if y<one/2 else one/4]) for x,y in zip(pls[k][b],pus[k][b])]
            for a in range(na):
                if na==1:row.policy_lipschitz[a]=z;continue
                pair=[]
                for j in range(na):
                    if j==a:continue
                    pair.extend(tu[v][1][b][a]-tl[v][1][b][j]-tl[v][0][b][a]+tu[v][0][b][j] for v in range(nv))
                    pair.extend(-tl[v][1][b][a]+tu[v][1][b][j]+tu[v][0][b][a]-tl[v][0][b][j] for v in range(nv))
                row.policy_lipschitz[a]=row.gibbs_factor[a]/eps*M.maximum(pair) if anchored else M.minimum([row.policy_lipschitz[a],row.gibbs_factor[a]/eps*M.maximum(pair)])
            forces=[]
            for v,(i,j) in enumerate(bg.edges):
                fl=[[arb(0) for _ in range(na)] for _ in range(2)]
                fu=[[arb(0) for _ in range(na)] for _ in range(2)]
                for a in range(1,na):
                    for s in range(2):
                        D=w[s][a][i]-w[s][a][j];E=g[s][a][i]-g[s][a][j]
                        assert D>0
                        fl[s][a]=state.eta*(E+D*(cl[s][b-a]-cu[s][b]))
                        fu[s][a]=state.eta*(E+D*(cu[s][b-a]-cl[s][b]))
                forces.append((fl,fu))
                lows=[M.minimum(fl[s][a] for s in range(2)) for a in range(na)]
                highs=[M.maximum(fu[s][a] for s in range(2)) for a in range(na)]
                avlo=M.box_min(lows,pls[k][b],pus[k][b]);avhi=M.box_max(highs,pls[k][b],pus[k][b])
                oldlo,oldhi=row.average_forcing[v]
                row.average_forcing[v]=(avlo,avhi) if anchored else (M.maximum([oldlo,avlo]),M.minimum([oldhi,avhi]))
            for a in range(na):
                if na==1:continue
                sep=M.maximum(v for fl,fu in forces for j in range(na) if j!=a for s in range(2) for v in [fu[s][a]-fl[s][j],fu[s][j]-fl[s][a]])
                row.instant_gain[a]=row.gibbs_factor[a]/eps*sep if anchored else M.minimum([row.instant_gain[a],row.gibbs_factor[a]/eps*sep])
        c.forcing_span=[M.maximum(M.maximum(c.row[i].average_forcing[v][1] for i in range(b+1))-M.minimum(c.row[i].average_forcing[v][0] for i in range(b+1)) for v in range(3)) for b in range(5)]
        c.spend_one_or_more=M.maximum(M.box_max([z]+[state.a[a].hi for a in range(1,len(r.policy_lower))],r.policy_lower,r.policy_upper) for r in c.row)
        c.spend_two=M.maximum([z]+[r.policy_upper[2]*state.a[2].hi for r in c.row if len(r.policy_lower)==3])
        # Each row's nonzero-bid mass includes its bid-2 mass.
        c.spend_one_or_more=M.maximum([c.spend_one_or_more,c.spend_two])
    mu1=mu2=z
    for c in state.c:
        mu1+=lam*dt*(c.spend_one_or_more-c.spend_two);mu2+=lam*dt*c.spend_two
        h0=(-mu1-mu2).exp()
        ps=[h0,h0*mu1,h0*(mu1**2/2+mu2),h0*(mu1**3/6+mu1*mu2)]
        c.probs=ps+[one-sum(ps,z)]
    return state,{'conditional_lower_t0':[[x.str(18) for x in r] for r in rightL],
        'conditional_upper_t0':[[x.str(18) for x in r] for r in rightU]}

def restrict_run(n,rounds,steps,out,cfg):
    box=([0,0,0],[1000000]*3);boxes=[box]
    for k in range(steps):
        st=bg.model(cfg.sources,cfg.shares,n,True,box);st,_=improve(st,box,cfg,rounds)
        lo,hi=bg.restrict(st)
        box=([max(x,y) for x,y in zip(box[0],lo)],[min(x,y) for x,y in zip(box[1],hi)])
        boxes.append(box);print('outer',k+1,box,flush=True)
    result={**metadata(cfg),'bits':ctx.prec,'cells':n,'conditional_iterations':rounds,'boxes':boxes,'scope':'every exact continuous-game fixed point lies in every nested box'}
    Path(out).write_text(json.dumps(result,indent=2)+'\n')

def verify_run(n,rounds,boxfile,out,cfg):
    record=json.loads(Path(boxfile).read_text())
    assert all(record[k]==json.loads(json.dumps(v)) for k,v in metadata(cfg).items())
    box=record['boxes'][-1]
    print('building',n,box,flush=True)
    st=bg.model(cfg.sources,cfg.shares,n,True,box);st,diag=improve(st,box,cfg,rounds)
    print('kernel enclosure',flush=True);K=bg.kernels(st)
    fs=bg.model(cfg.sources,cfg.shares,n,False,box)
    # Scalar search witness; exact integer weights are replayed below with Arb.
    for a,b in zip(fs.c,st.c):
        a.forcing_span=list(map(float,b.forcing_span));a.probs=list(map(float,b.probs))
        for x,y in zip(a.row,b.row):
            x.instant_gain=list(map(float,y.instant_gain));x.gibbs_factor=list(map(float,y.gibbs_factor))
    fk=[[[([float(v) for v in a],[float(v) for v in b]) for a,b in rr] for rr in R] for R in K]
    weight=np.ones(n)
    for _ in range(75):
        v=np.array(bg.evalw(fs,fk,weight));weight=v/v.max()
    wi=[int(round(v*SCALE)) for v in weight];assert min(wi)>0
    weights=[arb(v)/SCALE for v in wi]
    ys=bg.evalw(st,K,weights)
    ratios=[y/w for y,w in zip(ys,weights)];gain=st.math_ops.maximum(ratios)
    unweighted=st.math_ops.maximum(bg.evalw(st,K,[arb(1)]*n))
    result={**metadata(cfg),'bits':ctx.prec,'cells':n,'conditional_iterations':rounds,'box':box,'weight_scale':SCALE,'weight_integers':wi,
      'gain':gain.str(30),'gain_upper_decimal':str(rounded(gain,True)/SCALE),'strictly_below_one':bool(gain<1),
      'unweighted':unweighted.str(25),'ratio_upper_integers':[rounded(x,True) for x in ratios],
      'diagnostics':diag,'scope':'exact continuous-time continuous-belief population response; conditional on proved envelope and Duhamel lemmas'}
    def up(x):
        # These ledger entries bound nonnegative probabilities, seminorms and gains.
        # An outward ball for an exact zero may straddle zero.
        assert x.upper()>=0
        return max(0,rounded(x,True))
    lcells=[dict(prob=[up(x) for x in c.probs],source_span=[up(x) for x in c.forcing_span],
       direct=[[up(x) for x in r.instant_gain] for r in c.row],gibbs_slope=[[up(x) for x in r.gibbs_factor] for r in c.row]) for c in st.c]
    lkernels=[[[[[up(x) for x in a],[up(x) for x in b]] for a,b in rows] for rows in terminal] for terminal in K]
    ledger=dict(n=n,scale=SCALE,weight_scale=SCALE,weights=wi,cells=lcells,kernels=lkernels,box=box,**metadata(cfg))
    ledgerfile=str(Path(out).with_suffix('.ledger.json.gz'))
    Path(ledgerfile).write_bytes(gzip.compress(json.dumps(ledger,separators=(',',':')).encode(),mtime=0))
    from replay_weighted_integer import bound
    integer=bound(ledger);assert arb(integer)/SCALE>=gain, "upper ledger must dominate Arb aggregation"
    result['integer_aggregation_numerator']=integer;result['integer_aggregation_denominator']=SCALE
    result['integer_aggregation_below_one']=integer<SCALE
    result['ledger_sha256']=hashlib.sha256(Path(ledgerfile).read_bytes()).hexdigest()
    result['python_version']=platform.python_version()
    result['input_sha256']={name:hashlib.sha256((ROOT/'v5/code/sharp_certificates/comparison/inputs'/name).read_bytes()).hexdigest() for name in [cfg.sources+'.json','auction_fit.json']}
    Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['gain','strictly_below_one','unweighted']}),flush=True)
    if not (gain<1 and integer<SCALE):
        raise RuntimeError('The sufficient contraction test is inconclusive at these settings.')

if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outdir',type=Path)
    ap.add_argument('--eta',nargs=2,required=True,metavar=('LOW','HIGH'))
    ap.add_argument('--primitives',choices=('archived','refit','attributed','tail4h','pooled'),default='archived')
    ap.add_argument('--policy',choices=('shannon','anchored'),default='shannon')
    ap.add_argument('--cells',type=int,default=160)
    ap.add_argument('--bits',type=int,default=160)
    ap.add_argument('--epsilon',default='.05')
    a=ap.parse_args()
    assert 0<=Fraction(a.eta[0])<=Fraction(a.eta[1])<=1 and Fraction(a.epsilon)>0
    os.environ['EPS']=a.epsilon
    ctx.prec=a.bits
    a.outdir.mkdir(parents=True,exist_ok=True)
    cfg=Settings(a.primitives,tuple(dict.fromkeys(a.eta)),a.policy)
    restrict_run(40,6,8,a.outdir/'boxes.json',cfg)
    verify_run(a.cells,8,a.outdir/'boxes.json',a.outdir/'gain.json',cfg)
