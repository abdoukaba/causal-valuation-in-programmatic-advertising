#!/usr/bin/env python3
"""Independent end-to-end interval replay for the frozen composite auction game.

Independent implementation based on formulas in Supplemental 2, Appendix D,
NOT importing conditional_arb.py, budget_gain.py, enclose.py or Arb.

Uses mpmath.iv (libmpi, directed binary interval endpoints) and exact decimal
input parsing.  Candidate weights are untrusted rationals: the final verified
inequality is recomputed from primitive inputs. The original ledger is not read.

Proof scope is the analytical envelope/positive-comparison inequalities stated
in the supplement, not a new formal verification of those analytic lemmas.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, time
from functools import lru_cache
from pathlib import Path
from mpmath import iv, libmp

ROOT=Path(__file__).parent
BASE=ROOT/'data'
INPUTS=BASE
S=10**12
SMALL=10**6
iv.dps=65

def I(x):
    if isinstance(x,int): return iv.mpf(x)
    return iv.mpf(str(x))
Z=I(0); U=I(1); HALF=I('0.5')
RATE=I('2.25'); EPS=I('0.05'); ETA=I('0.45')
CMP=libmp.mpf_cmp

def hull_extrema(xs, mode):
    it=iter(xs)
    try: first=next(it)
    except StopIteration: raise ValueError('empty extrema')
    low,high=first._mpi_
    for v in it:
        x,y=v._mpi_
        if (CMP(x,low)>0 if mode=='max' else CMP(x,low)<0):low=x
        if (CMP(y,high)>0 if mode=='max' else CMP(y,high)<0):high=y
    return iv.make_mpf((low,high))

def hi(xs): return hull_extrema(xs,'max')
def lo(xs): return hull_extrema(xs,'min')
def contains_zero(x):
    l,h=x._mpi_
    return CMP(l,libmp.fzero)<=0<=CMP(h,libmp.fzero)
def exact_bound(x,which):
    b=x._mpi_[1 if which=='upper' else 0]
    s,m,e,_=b
    n=-m if s else m
    return (n<<e,1) if e>=0 else (n,1<<-e)
def floor_scaled(x,mult):
    n,d=exact_bound(x,'lower'); return n*mult//d

def ceil_scaled(x,mult):
    n,d=exact_bound(x,'upper'); return -(-n*mult//d)

def certain_lt(x,y): return CMP(x._mpi_[1],y._mpi_[0])<0

def box_minmax(vals,lows,ups,maximum=True):
    """LP dual for a box intersected with the simplex; interval arithmetic."""
    answers=[]
    for c in vals:
        acc=c
        for v,l,u in zip(vals,lows,ups):
            diff=v-c
            acc+=u*hi((diff,Z))+l*lo((diff,Z))
        answers.append(acc)
    result=lo(answers)  # dual min of supporting price candidates
    return result if maximum else -box_minmax([-v for v in vals],lows,ups,True)


def simplex_vertices(L,H,denom):
    pts=set()
    d=len(L)
    for free in range(d):
        remaining=[j for j in range(d) if j!=free]
        for bits in itertools.product((0,1),repeat=d-1):
            v=[0]*d
            for j,b in zip(remaining,bits):v[j]=H[j] if b else L[j]
            v[free]=denom-sum(v)
            if L[free]<=v[free]<=H[free]:pts.add(tuple(v))
    if not pts:raise ValueError(f'infeasible simplex L={L} H={H}')
    return sorted(pts)

@lru_cache(maxsize=300000)
def entropy_at_vertex(v):
    prob=[I(x)/S for x in v]
    return -sum((p*iv.ln(p) for p,x in zip(prob,v) if x),Z)

@lru_cache(maxsize=200000)
def rational_policy_vertices(L,H):
    return tuple(simplex_vertices(L,H,S))

def policy_vertices(policy_l,policy_u):
    L=tuple(max(0,floor_scaled(x,S)) for x in policy_l)
    H=tuple(min(S,ceil_scaled(x,S)) for x in policy_u)
    assert sum(L)<=S<=sum(H) and all(l<=h for l,h in zip(L,H))
    return rational_policy_vertices(L,H)

def entropy_stage_lower(scores,vertex_pis):
    return lo((sum((I(x)/S*y for x,y in zip(v,scores)),Z)+EPS*entropy_at_vertex(v) for v in vertex_pis))

def lse_upper(scores):
    return EPS*iv.ln(sum((iv.exp(a/EPS) for a in scores),Z))

def abs_span(seq):
    v=list(seq)
    return hi((abs(a-b) for a in v for b in v))

class Replay:
    def __init__(self, bits_dps=65, chatty=True):
        iv.dps=bits_dps;self.chatty=chatty
        self.grid=[I(x) for x in ['0','0.05','0.10','0.15']]
        def get(path):return json.loads(path.read_text(),parse_float=str)
        pr=get(INPUTS/'archived.json')['criteo_offsets']['primitives']
        ft=get(INPUTS/'auction_fit.json')['top=450']
        assert pr['q0']==ft['q0'] and pr['zeta']==ft['zeta_fit']
        self.q0=[I(x) for x in pr['q0']]
        self.D=[[I(x) for x in v] for v in pr['Delta']]
        self.d=[I(x) for x in pr['d']]
        self.zeta=I(pr['zeta'])
        self.w=[[[Z for _ in range(4)] for _ in range(3)] for _ in range(2)]
        self.g=[[[Z for _ in range(4)] for _ in range(3)] for _ in range(2)]
        for r in range(2):
            for a in (1,2):
                for m in range(4):
                    self.w[r][a][m]=U/(U+iv.exp(-(self.grid[a]-self.grid[m]-self.d[r])/self.zeta))
                    self.g[r][a][m]=self.w[r][a][m]*(self.D[r][m]-self.grid[a])
        for r in range(2):
            for a in (1,2):
                assert all(certain_lt(Z,x) and certain_lt(x,U) for x in self.w[r][a])
        self.setup_cache={}

    def setup(self,box):
        key=(tuple(box[0]),tuple(box[1]))
        if key in self.setup_cache:return self.setup_cache[key]
        verts=simplex_vertices(box[0],box[1],SMALL)
        qverts=[]
        for v in verts:
            qverts.append([(U-ETA)*self.q0[j]+(ETA*I(v[j])/SMALL if j<3 else Z) for j in range(4)])
        ell=[[[sum((q[j]*self.w[s][a][j] for j in range(4)),Z) for a in range(3)] for s in range(2)] for q in qverts]
        reward=[[[sum((q[j]*self.g[s][a][j] for j in range(4)),Z) for a in range(3)] for s in range(2)] for q in qverts]
        for arr in ell:
            for s in range(2):
                for a in (1,2):assert certain_lt(Z,arr[s][a]) and certain_lt(arr[s][a],U)
        out=(ell,reward)
        self.setup_cache[key]=out
        return out

    def bayes_kernel_tests(self,box):
        """Interval Bayes maps, their derivatives, tower identity and conditioning."""
        ell,_=self.setup(box)
        # All quantities at each simplex vertex. Uniform coverage follows
        # from positive affine likelihoods and linear fractional Bayes ratios.
        n=0; win_min=U; loss_min=U; identity_worst=0
        for Q in ell:
            for a in (1,2):
                e0,e1=Q[0][a],Q[1][a]
                win_min=lo([win_min,e0,e1]);loss_min=lo([loss_min,U-e0,U-e1])
                for p_string in ('0','0.01','0.1','0.5','0.9','0.99','1'):
                    p=I(p_string)
                    W=(U-p)*e0+p*e1
                    Pwin=p*e1/W
                    Ploss=p*(U-e1)/(U-W)
                    # Taking interval intersections with [0,1] is justified
                    # analytically by positive conditional probabilities.
                    assert CMP(Pwin._mpi_[1],U._mpi_[0])<=0 or CMP(Pwin._mpi_[0],U._mpi_[1])<=0
                    assert CMP(Ploss._mpi_[0],Z._mpi_[1])>=0 or CMP(Ploss._mpi_[1],Z._mpi_[0])>=0
                    # avoid subtraction cancellation by the equivalent numerator identity
                    posterior_avg=W*Pwin+(U-W)*Ploss
                    delta=posterior_avg-p
                    assert contains_zero(delta)
                    dw=e0*e1/(W*W)
                    dl=(U-e0)*(U-e1)/((U-W)*(U-W))
                    assert certain_lt(Z,dw) and certain_lt(Z,dl)
                    n+=1
        return {'bayes_checks':n,'positive_win_lower':fmt(win_min),'positive_loss_lower':fmt(loss_min),'martingale_identity_checked':True,'belief_mesh_used':False}

    def initial_action_bounds(self,box):
        ell,rew=self.setup(box)
        out=[None]
        for a in (1,2):
            all_w=[v[s][a] for v in ell for s in range(2)]
            all_g=[v[s][a] for v in rew for s in range(2)]
            Lg=abs_span(all_g)
            Lw=abs_span(all_w)
            sep=Z
            for s in (0,1):
                ratio=hi((v[s][a]/v[1-s][a] for v in ell))
                lossratio=hi(((U-v[1-s][a])/(U-v[s][a]) for v in ell))
                root=iv.sqrt(ratio*lossratio)
                sep=hi((sep,(root-U)/(root+U)))
            edges=[]
            for i,j in ((0,1),(0,2),(1,2)):
                dg=[self.g[s][a][i]-self.g[s][a][j] for s in (0,1)]
                dw=[self.w[s][a][i]-self.w[s][a][j] for s in (0,1)]
                assert all(certain_lt(Z,x) for x in dw)
                beta=Z
                for comp in (False,True):
                    pairs=[((U-v[0][a],U-v[1][a]) if comp else (v[0][a],v[1][a])) for v in ell]
                    denom=(iv.sqrt(lo(x for x,y in pairs))+iv.sqrt(lo(y for x,y in pairs)))**2
                    nom=hi((abs(x*dw[1]-y*dw[0]) for x,y in pairs))
                    beta+=nom/denom
                edges.append((dg,dw,beta))
            out.append({'lo':lo(all_w),'hi':hi(all_w),'Lg':Lg,'Lw':Lw,'sep':sep,'edges':edges,'all_reward':all_g,'all_likelihood':all_w})
        return out

    def baseline(self,box,n):
        """Independent restatement of continuation-gap and Gibbs inequalities."""
        act=self.initial_action_bounds(box)
        dt=U/n
        survive=iv.exp(-RATE*dt);hit=U-survive
        R=EPS*iv.ln(U+sum((iv.exp(hi(act[i]['all_reward'])/EPS) for i in (1,2)),Z))
        at_right_l=[Z]*5;at_right_u=[Z]*5
        lip=Z
        rows=[None]*n
        for k in range(n-1,-1,-1):
            in_lower=[survive*v for v in at_right_l]
            in_upper=[hi(at_right_u[:b+1])+RATE*dt*R if b else Z for b in range(5)]
            gaps=[]
            for b in range(5):
                gap=[]
                for a in range(min(2,b)+1):
                    gap.append((hi((Z,in_lower[b]-in_upper[b-a])),hi((Z,in_upper[b]-in_lower[b-a]))))
                gaps.append(gap)
            lip+=RATE*dt*hi((act[a]['Lg']+act[a]['Lw']*gaps[b][a][1] for b in range(1,5) for a in range(1,min(2,b)+1)))
            left_l=[];left_u=[];cell=[]
            for b in range(5):
                actions=range(min(2,b)+1)
                cent_l=[Z];cent_u=[Z]
                scores_l=[in_lower[b]];scores_u=[in_upper[b]]
                for a in list(actions)[1:]:
                    g=act[a]['all_reward'];w=act[a]['all_likelihood'];gapl,gapu=gaps[b][a]
                    cent_l.append(lo((x-y*gapu for x,y in zip(g,w))))
                    cent_u.append(hi((x-y*gapl for x,y in zip(g,w)))+lip*act[a]['Lw']/2)
                    scores_l.append(lo((x+(U-y)*in_lower[b]+y*in_lower[b-a] for x,y in zip(g,w))))
                    scores_u.append(hi((x+(U-y)*in_upper[b]+y*in_upper[b-a] for x,y in zip(g,w))))
                left_l.append(survive*at_right_l[b]+hit*lse_upper(scores_l))
                left_u.append(survive*at_right_u[b]+hit*lse_upper(scores_u))
                pl=[];pu=[]
                for a in actions:
                    pl.append(U/(U+sum((iv.exp((cent_u[j]-cent_l[a])/EPS) for j in actions if j!=a),Z)))
                    pu.append(U/(U+sum((iv.exp((cent_l[j]-cent_u[a])/EPS) for j in actions if j!=a),Z)))
                factors=[self.factor(x,y) for x,y in zip(pl,pu)]
                score_lip=[lip]+[lip+act[a]['Lg']+act[a]['Lw']*gaps[b][a][1] for a in list(actions)[1:]]
                plips=[factors[a]*(score_lip[a]+hi((score_lip[j] for j in actions if j!=a)))/EPS if len(pl)>1 else Z for a in actions]
                avg=[];inst=[Z]*len(pl)
                for edge in range(3):
                    FL=[Z];FU=[Z]
                    for a in list(actions)[1:]:
                        g,w,beta=act[a]['edges'][edge]
                        gapl,gapu=gaps[b][a]
                        swing=lip*(hi(w)*act[a]['sep']+beta)
                        FL.append(ETA*(lo((gs-ws*gapu for gs,ws in zip(g,w)))-swing))
                        FU.append(ETA*(hi((gs-ws*gapl for gs,ws in zip(g,w)))+swing))
                    for a in actions:
                        if len(pl)>1:
                            v1=FU[a]-lo((FL[j] for j in actions if j!=a))
                            v2=hi((FU[j] for j in actions if j!=a))-FL[a]
                            inst[a]=hi((inst[a],factors[a]/EPS*hi((v1,v2))))
                    avg.append((box_minmax(FL,pl,pu,False),box_minmax(FU,pl,pu,True)))
                cell.append({'pl':pl,'pu':pu,'factor':factors,'plip':plips,'avg':avg,'instant':inst})
            at_right_l,at_right_u=left_l,left_u
            c={'row':cell}
            self.cell_coefficients(c,act)
            rows[k]=c
        self.probabilities(rows,dt)
        return {'c':rows,'act':act,'n':n,'dt':dt,'survive':survive,'hit':hit}

    @staticmethod
    def factor(x,y):
        # Upper envelope for pi_a(1-pi_a) on proven pi in [x,y].
        vals=[U/4]
        if CMP(x._mpi_[0],HALF._mpi_[1])>0: vals.append(x*(U-x))
        if CMP(y._mpi_[1],HALF._mpi_[0])<0: vals.append(y*(U-y))
        return lo(vals)

    @staticmethod
    def cell_coefficients(c,act):
        c['forcing_span']=[hi((hi(c['row'][i]['avg'][edge][1] for i in range(b+1))-lo(c['row'][i]['avg'][edge][0] for i in range(b+1)) for edge in range(3))) for b in range(5)]
        one=Z;two=Z
        for r in c['row']:
            actions=range(len(r['pl']))
            score=[Z]+[act[a]['hi'] for a in list(actions)[1:]]
            one=hi((one,box_minmax(score,r['pl'],r['pu'],True)))
            if len(r['pl'])==3: two=hi((two,r['pu'][2]*act[2]['hi']))
        c['spend_two']=two
        c['spend_one']=hi((one,two))
        assert certain_lt(c['spend_two'],U) and certain_lt(c['spend_one'],U)

    @staticmethod
    def probabilities(cells,dt):
        a=b=Z
        for c in cells:
            a+=RATE*dt*(c['spend_one']-c['spend_two'])
            b+=RATE*dt*c['spend_two']
            # A potentially wide interval for a-b is fine because the
            # original inequality proves it is nonnegative.
            h0=iv.exp(-a-b)
            p=[h0,h0*a,h0*(a*a/2+b),h0*(a*a*a/6+a*b)]
            c['probs']=p+[U-sum(p,Z)]
            assert all(not certain_lt(x,Z) for x in c['probs'])

    def improved(self,st,box,rounds):
        ell,reward=self.setup(box)
        NV=len(ell);n=st['n'];dt=st['dt'];sv=st['survive'];hit=st['hit']
        acts=st['act']
        pls=[[list(r['pl']) for r in c['row']] for c in st['c']]
        pus=[[list(r['pu']) for r in c['row']] for c in st['c']]
        zeros=lambda:[[Z for _ in range(5)] for _ in range(2)]
        last=None
        for iteration in range(rounds):
            rightL=zeros();rightU=zeros();cells=[None]*n
            for k in range(n-1,-1,-1):
                cellL=zeros();cellU=zeros()
                pvs=[policy_vertices(pls[k][b],pus[k][b]) for b in range(5)]
                for s in range(2):
                    stage_lo=Z
                    for b in range(1,5):
                        na=min(2,b)+1
                        stage_lo=lo([stage_lo]+[entropy_stage_lower(reward[v][s][:na],pvs[b]) for v in range(NV)])
                        upp=hi((lse_upper(reward[v][s][:na]) for v in range(NV)))
                        cellL[s][b]=sv*rightL[s][b]+hit*lo(rightL[s][:b+1])+RATE*dt*stage_lo
                        cellU[s][b]=sv*rightU[s][b]+hit*hi(rightU[s][:b+1])+RATE*dt*hi((Z,upp))
                TL=[[[[Z for _ in range(3)] for _ in range(5)] for _ in range(2)] for _ in range(NV)]
                TU=[[[[Z for _ in range(3)] for _ in range(5)] for _ in range(2)] for _ in range(NV)]
                for b in range(5):
                    actions=range(min(2,b)+1)
                    for a in actions:
                        for v in range(NV):
                            for s in (0,1):
                                w=ell[v][s][a];r=reward[v][s][a]
                                TL[v][s][b][a]=r+(U-w)*cellL[s][b]+w*cellL[s][b-a]
                                TU[v][s][b][a]=r+(U-w)*cellU[s][b]+w*cellU[s][b-a]
                    for a in actions:
                        dl=[Z if j==a else lo((TL[v][s][b][j]-TU[v][s][b][a] for v in range(NV) for s in (0,1))) for j in actions]
                        du=[Z if j==a else hi((TU[v][s][b][j]-TL[v][s][b][a] for v in range(NV) for s in (0,1))) for j in actions]
                        pls[k][b][a]=hi((pls[k][b][a],U/sum((iv.exp(x/EPS) for x in du),Z)))
                        pus[k][b][a]=lo((pus[k][b][a],U/sum((iv.exp(x/EPS) for x in dl),Z)))
                    pvs[b]=policy_vertices(pls[k][b],pus[k][b])
                leftL=zeros();leftU=zeros()
                for s in (0,1):
                    for b in range(1,5):
                        na=min(2,b)+1
                        leftL[s][b]=sv*rightL[s][b]+hit*lo((entropy_stage_lower(TL[v][s][b][:na],pvs[b]) for v in range(NV)))
                        leftU[s][b]=sv*rightU[s][b]+hit*hi((lse_upper(TU[v][s][b][:na]) for v in range(NV)))
                cells[k]=(cellL,cellU,TL,TU)
                rightL,rightU=leftL,leftU
            last=cells
            if self.chatty: print('conditional pass',iteration+1,'of',rounds,'cells',n,flush=True)
        for k,c in enumerate(st['c']):
            cellL,cellU,TL,TU=last[k]
            for b,row in enumerate(c['row']):
                actions=range(min(2,b)+1)
                row['pl']=pls[k][b];row['pu']=pus[k][b]
                row['factor']=[self.factor(x,y) for x,y in zip(row['pl'],row['pu'])]
                for a in actions:
                    if len(row['pl'])==1:
                        row['plip'][a]=Z; continue
                    pair=[]
                    for j in actions:
                        if j==a:continue
                        for v in range(NV):
                            pair.append(TU[v][1][b][a]-TL[v][1][b][j]-TL[v][0][b][a]+TU[v][0][b][j])
                            pair.append(-TL[v][1][b][a]+TU[v][1][b][j]+TU[v][0][b][a]-TL[v][0][b][j])
                    row['plip'][a]=lo((row['plip'][a],row['factor'][a]/EPS*hi(pair)))
                all_fl=[];all_fu=[]
                for edge,(i,j) in enumerate(((0,1),(0,2),(1,2))):
                    FL=[[Z]*len(row['pl']) for _ in range(2)]
                    FU=[[Z]*len(row['pl']) for _ in range(2)]
                    for a in list(actions)[1:]:
                        for s in (0,1):
                            D=self.w[s][a][i]-self.w[s][a][j]
                            E=self.g[s][a][i]-self.g[s][a][j]
                            assert certain_lt(Z,D)
                            FL[s][a]=ETA*(E+D*(cellL[s][b-a]-cellU[s][b]))
                            FU[s][a]=ETA*(E+D*(cellU[s][b-a]-cellL[s][b]))
                    all_fl.append(FL);all_fu.append(FU)
                    low=[lo((FL[s][a] for s in (0,1))) for a in actions]
                    upper=[hi((FU[s][a] for s in (0,1))) for a in actions]
                    avglo=box_minmax(low,row['pl'],row['pu'],False)
                    avghi=box_minmax(upper,row['pl'],row['pu'],True)
                    oldlo,oldhi=row['avg'][edge]
                    row['avg'][edge]=(hi((oldlo,avglo)),lo((oldhi,avghi)))
                for a in actions:
                    if len(row['pl'])==1:continue
                    spans=[]
                    for edge in range(3):
                        for j in actions:
                            if j==a:continue
                            for s in (0,1):
                                spans.append(all_fu[edge][s][a]-all_fl[edge][s][j])
                                spans.append(all_fu[edge][s][j]-all_fl[edge][s][a])
                    row['instant'][a]=lo((row['instant'][a],row['factor'][a]/EPS*hi(spans)))
            self.cell_coefficients(c,acts)
        self.probabilities(st['c'],dt)
        return st,[[fmt(x) for x in row] for row in rightL],[[fmt(x) for x in row] for row in rightU]

    @staticmethod
    def occ(values,p):
        return sum((p[k]*hi((values[b] for b in range(4-k,5))) for k in range(5)),Z)

    def restrict(self,st):
        low=[];upper=[]
        for j in range(3):
            a=[];b=[]
            for c in st['c']:
                lows=[-(r['pl'][j] if len(r['pl'])>j else Z) for r in c['row']]
                highs=[r['pu'][j] if len(r['pu'])>j else Z for r in c['row']]
                a.append(-self.occ(lows,c['probs']))
                b.append(self.occ(highs,c['probs']))
            lower=lo(a);higher=hi(b)
            ll=max(0,floor_scaled(lower,SMALL)-2)
            uu=min(SMALL,ceil_scaled(higher,SMALL)+2)
            assert (ll==0 or not certain_lt(lower,I(ll)/SMALL))
            assert (uu==SMALL or not certain_lt(I(uu)/SMALL,higher))
            low.append(ll);upper.append(uu)
        return low,upper

    def outer_boxes(self,n=40,steps=8,rounds=6):
        box=([0,0,0],[SMALL]*3)
        all_boxes=[box]
        for k in range(steps):
            st=self.baseline(box,n)
            st,_,_=self.improved(st,box,rounds)
            low,high=self.restrict(st)
            box=([max(a,b) for a,b in zip(box[0],low)],[min(a,b) for a,b in zip(box[1],high)])
            all_boxes.append(box)
            print('outer step',k+1,'box',box,flush=True)
        return all_boxes

    def kernel_bounds(self,st):
        C=st['c'];N=st['n'];dt=st['dt'];sv=st['survive'];hit=st['hit'];act=st['act']
        K=[]
        for t in range(N):
            terminal=[]
            for j in range(3):
                lower=[row['pl'][j] if len(row['pl'])>j else Z for row in C[t]['row']]
                upper=[row['pu'][j] if len(row['pu'])>j else Z for row in C[t]['row']]
                lip=hi([Z]+[row['plip'][j] for row in C[t]['row'] if len(row['plip'])>j])
                block=[]
                for s in range(t,-1,-1):
                    cell=C[s]
                    l=[sv*v+hit*lo(lower[:b+1]) for b,v in enumerate(lower)]
                    u=[sv*v+hit*hi(upper[:b+1]) for b,v in enumerate(upper)]
                    spans=[];cross=[];growth=[];newL=[];newU=[]
                    for b,row in enumerate(cell['row']):
                        actions=range(len(row['pl']))
                        kmin=[l[b]];kmax=[u[b]];gaps=[Z]
                        for a in list(actions)[1:]:
                            W=act[a]
                            kmin.append(lo(((U-w)*l[b]+w*l[b-a] for w in (W['lo'],W['hi']))))
                            kmax.append(hi(((U-w)*u[b]+w*u[b-a] for w in (W['lo'],W['hi']))))
                            gaps.append(hi((u[b]-l[b-a],u[b-a]-l[b])))
                        spread=hi([Z]+[kmax[a]-kmin[z] for a in actions for z in actions if z!=a])
                        spans.append(spread);cross.append(gaps)
                        growth.append(box_minmax([Z]+[act[a]['Lw']*gaps[a] for a in list(actions)[1:]],row['pl'],row['pu'],True)+hi(row['plip'])*spread)
                        Lc=sv*lower[b]+hit*box_minmax(kmin,row['pl'],row['pu'],False)
                        Uc=sv*upper[b]+hit*box_minmax(kmax,row['pl'],row['pu'],True)
                        if s==t:Lc=lo((Lc,lower[b])); Uc=hi((Uc,upper[b]))
                        newL.append(Lc);newU.append(Uc)
                    lip+=RATE*dt*hi(growth)
                    env=[]
                    for b,row in enumerate(cell['row']):
                        vv=[]
                        for edge in range(3):
                            amounts=[Z]
                            for a in range(1,len(row['pl'])):
                                dg,dw,beta=act[a]['edges'][edge]
                                amounts.append(hi(dw)*cross[b][a]+beta*lip)
                            vv.append(box_minmax(amounts,row['pl'],row['pu'],True))
                        env.append(ETA*hi(vv))
                    block.append((spans,env))
                    lower,upper=newL,newU
                terminal.append(block[::-1])
            K.append(terminal)
            if self.chatty and (t+1)%40==0:print('test kernels',t+1,'of',N,flush=True)
        return K

    def gain(self,st,K,weight_integers):
        N=st['n'];dt=st['dt'];C=st['c']
        weights=[I(v)/S for v in weight_integers]
        assert len(weights)==N and all(certain_lt(Z,w) for w in weights)
        future=[Z]*5;past=[None]*N;coord=[None]*N
        for k in range(N-1,-1,-1):
            cell=C[k]
            future=[f+RATE*dt*span*weights[k] for f,span in zip(future,cell['forcing_span'])]
            coord[k]=[[v*weights[k]+g/EPS*future[b] for v,g in zip(row['instant'],row['factor'])] for b,row in enumerate(cell['row'])]
            past[k]=[hi(row) for row in coord[k]]
        gain=[]
        for t in range(N):
            values=[]
            for j in range(3):
                direct=self.occ([(row[j] if len(row)>j else Z) for row in coord[t]],C[t]['probs'])
                for s,(sp,ev) in enumerate(K[t][j]):
                    direct+=RATE*dt*self.occ([a*P+b*weights[s] for a,P,b in zip(sp,past[s],ev)],C[s]['probs'])
                values.append(direct)
            gain.append(hi(values)/weights[t])
            if self.chatty and (t+1)%40==0:print('gain',t+1,'of',N,flush=True)
        return hi(gain),gain

def fmt(x,digits=18):
    return iv.nstr(x,digits)

def pick_weights(n):
    # Weights are fixed rational search witnesses, not bound evidence.
    d=json.loads((BASE/f'weights_{n}.json').read_text())
    assert d['n']==n and d['scale']==S and len(d['weights'])==n
    assert min(d['weights'])>0 and max(d['weights'])==S
    return d['weights']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cells',type=int,default=160)
    p.add_argument('--dps',type=int,default=55)
    p.add_argument('--steps',type=int,default=0,help='With --steps 8, recompute all outer boxes in this same process')
    p.add_argument('--boxfile',default=str(BASE/'outer_boxes_independent.json'),help='Independent completed outer-box JSON file; no original ledger/box inputs are needed')
    p.add_argument('--rounds',type=int,default=8)
    p.add_argument('--out',default=str(ROOT/'independent_replay.json'))
    p.add_argument('--quiet',action='store_true')
    a=p.parse_args()
    start=time.monotonic()
    r=Replay(a.dps,not a.quiet)
    if a.steps:
        boxes=r.outer_boxes(40,a.steps,6)
        box=boxes[-1]
    else:
        bdata=json.loads(Path(a.boxfile).read_text())
        if not bdata.get('complete'):raise ValueError('Outer-box replay not complete')
        boxes=bdata['boxes']
        if len(boxes)!=9 or boxes[0]!=[[0,0,0],[SMALL,SMALL,SMALL]]:raise ValueError('Outer boxes must start at entire simplex')
        box=boxes[-1]
    bayes=r.bayes_kernel_tests(box)
    print('Bayesian maps checked',bayes,flush=True)
    st=r.baseline(box,a.cells)
    st,lower,upper=r.improved(st,box,a.rounds)
    K=r.kernel_bounds(st)
    weights=pick_weights(a.cells)
    ub,gains=r.gain(st,K,weights)
    uppernum=ceil_scaled(ub,S)
    result={
        'implementation':'Independent mpmath.iv interval arithmetic, no Arb imports',
        'interval_decimal_digits':a.dps,'cells':a.cells,'box':box,'outer_boxes_reconstructed':boxes,
        'bayesian':bayes,'conditional_lower_at_t0':lower,
        'conditional_upper_at_t0':upper,
        'verified_gain_interval':fmt(ub,28),
        'verified_upper_1e12':str(uppernum),
        'strict_contraction':uppernum<S,
        'candidate_weights_sha256':hashlib.sha256(json.dumps(weights).encode()).hexdigest(),
        'primitive_archive_sha256':hashlib.sha256((INPUTS/'archived.json').read_bytes()).hexdigest(),
        'primitive_fit_sha256':hashlib.sha256((INPUTS/'auction_fit.json').read_bytes()).hexdigest(),
        'elapsed_seconds':round(time.monotonic()-start,2),
        'limitations':'Independent interval replay of coded comparison inequalities, not a formal proof checker for the analytical lemmas or the interval library.'
    }
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('cells','verified_gain_interval','verified_upper_1e12','strict_contraction','elapsed_seconds')},indent=2),flush=True)
    if not result['strict_contraction']:raise SystemExit('Comparison did not certify strict contraction')

if __name__=='__main__':main()
