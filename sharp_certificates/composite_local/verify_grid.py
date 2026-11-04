"""Validated first-price, partially observed dynamic response using Arb ball arithmetic.

Tangent coordinates are (abstain,low); high = 1-abstain-low. Floating branch
guesses never enter a proof without separate signed Arb inequalities.
"""
import argparse
import json
import time
from pathlib import Path
from flint import arb as A, arb_mat as M, ctx

ROOT = Path(__file__).resolve().parent
ZERO, ONE = A(0), A(1)

def upper_decimal(x, places=12):
    scale = 10**places
    num = int((x.upper()*scale).upper().ceil().unique_fmpz())
    assert num >= 0
    return f'{num//scale}.{num%scale:0{places}d}'

def row_norm(matrix):
    return maxupper(sum(abs(matrix[i,j]) for j in range(matrix.ncols())) for i in range(matrix.nrows()))

def maxupper(seq):
    out = ZERO
    for item in seq: out=out.max(item)
    return out

def transweights(n, i, prob, high_joint, dprob, djoint, stats):
    """Probability times posterior P1 weights, and all possible cell slopes."""
    if i==0 or i==n-1: return {i:(prob,dprob)}
    assert prob>0
    z = (n-1)*high_joint
    seed = max(0,min(n-2,int(float((z/prob).mid()))))
    if z > seed*prob and z < (seed+1)*prob:
        active=[seed]
    else:
        active=[k for k in range(n-1) if not (z<k*prob or z>(k+1)*prob)]
    assert active
    stats['interior_events']+=1
    stats['multicell_events']+=int(len(active)>1)
    weights={}
    for j in range(min(active),max(active)+2):
        branches=[]
        for k in active:
            if j==k:
                branches.append(((k+1)*prob-z,[(k+1)*u-(n-1)*v for u,v in zip(dprob,djoint)]))
            elif j==k+1:
                branches.append((z-k*prob,[(n-1)*v-k*u for u,v in zip(dprob,djoint)]))
            else:branches.append((ZERO,[ZERO,ZERO]))
        w, ds=branches[0]
        for v,d in branches[1:]:
            w=w.union(v); ds=[x.union(y) for x,y in zip(ds,d)]
        weights[j]=(w.max(ZERO).min(prob),ds)
    return weights

class Grid:
    def __init__(self,n,nt):
        self.n,self.nt,self.s=n,nt,5*n
        self.jump=A(9)/(4*nt);self.eps=ONE/20
        raw=json.loads((ROOT/'data/archived.json').read_text(),parse_float=str)['criteo_offsets']['primitives']
        bg=list(map(A,raw['q0'])); eta=A(45)/100
        assert sum(bg).contains(1)
        self.prior=A(raw['p0']);self.prim={}
        self.stats=dict(interior_events=0,multicell_events=0)
        for a in (1,2):
            for s in (0,1):
                wins=[1/(1+((A(k-a)/20+A(raw['d'][s]))/A(raw['zeta'])).exp()) for k in range(4)]
                gain=[w*(A(v)-A(a)/20) for w,v in zip(wins,raw['Delta'][s])]
                for code,coeff in [('w',wins),('g',gain)]:
                    const=(1-eta)*sum(v*q for v,q in zip(coeff,bg))+eta*coeff[2]
                    self.prim[a,s,code]=(const,[eta*(coeff[i]-coeff[2]) for i in (0,1)])
        self.H=M(self.s,self.s)
        for row in range(self.s):
            i=row%n;p=A(i)/(n-1)
            u=(p*(1-p)*(n-1)/5)**2/(2*nt)
            stay=1-self.jump-2*u
            assert stay>0
            self.H[row,row]=stay
            if i>0:self.H[row,row-1]=u
            if i<n-1:self.H[row,row+1]=u

    def terms(self,x):
        n,S=self.n,self.s
        assert x[0]>0 and x[1]>0 and 1-x[0]-x[1]>0
        K=[M(S,S) for _ in range(3)]
        dK=[[M(S,S),M(S,S)] for _ in range(3)]
        gain=[M(S,1) for _ in range(3)]
        dg=[M(S,2) for _ in range(3)]
        for i in range(S):K[0][i,i]=1
        def at(a,s,code):
            c,d=self.prim[a,s,code]
            return c+sum(z*w for z,w in zip(x,d)),d
        for a in (1,2):
            w0,d0=at(a,0,'w');w1,d1=at(a,1,'w')
            g0,h0=at(a,0,'g');g1,h1=at(a,1,'g')
            for i in range(n):
                p=A(i)/(n-1);win=(1-p)*w0+p*w1
                dw=[(1-p)*u+p*v for u,v in zip(d0,d1)]
                for spend,prob,joint,dp,dj in [(a,win,p*w1,dw,[p*v for v in d1]),
                        (0,1-win,p*(1-w1),[-z for z in dw],[-p*v for v in d1])]:
                    weights=transweights(n,i,prob,joint,dp,dj,self.stats)
                    for b in range(a,5):
                        row=b*n+i
                        for j,(weight,dweight) in weights.items():
                            col=(b-spend)*n+j;K[a][row,col]=weight
                            for l in (0,1):dK[a][l][row,col]=dweight[l]
                for b in range(a,5):
                    row=b*n+i;gain[a][row,0]=(1-p)*g0+p*g1
                    for l in (0,1):dg[a][row,l]=(1-p)*h0[l]+p*h1[l]
        return K,dK,gain,dg

    def sweep(self,points,want_deriv):
        S,n,nt=self.s,self.n,self.nt
        v=M(S,1); d=2*nt if want_deriv else 0
        dv=M(S,d); stages=[]
        for t in reversed(range(nt)):
            K,dK,g,dg=self.terms(points[t])
            score=[K[a]*v+g[a] for a in range(3)]
            pi=M(S,3); h=M(S,1)
            for i in range(S):
                acts=range(min(i//n,2)+1)
                ex=[(score[a][i,0]/self.eps).exp() for a in acts];den=sum(ex)
                h[i,0]=self.eps*den.log()
                for a,e in zip(acts,ex):pi[i,a]=e/den
            if want_deriv:
                ds=[K[a]*dv for a in range(3)]
                for a in (1,2):
                    for l in (0,1):
                        tmp=dK[a][l]*v
                        for i in range(S):
                            ds[a][i,2*t+l]+=tmp[i,0]+dg[a][i,l]
                mean=M(S,d)
                # Rows after time t are strictly upper triangular in time.
                for i in range(S):
                    acts=range(min(i//n,2)+1)
                    for j in range(2*t,d):
                        mean[i,j]=sum(pi[i,a]*ds[a][i,j] for a in acts)
                newdv=self.H*dv+mean*self.jump
                for a in range(3):
                    for i in range(a*n,S):
                        fac=pi[i,a]/self.eps
                        for j in range(2*t,d):
                            ds[a][i,j]=fac*(ds[a][i,j]-mean[i,j])
                dv=newdv
            else:ds=None
            stages.append((pi,K,dK if want_deriv else None,ds))
            v=self.H*v+h*self.jump
        m=M(S,1);u=self.prior*(n-1);j=int(float(u.mid()))
        assert u>=j and u<=j+1
        m[4*n+j,0]=1-(u-j);m[4*n+j+1,0]=u-j
        dm=M(S,d);out=[];jac=M(2*nt,d)
        for t,(pi,K,dK,ds) in enumerate(reversed(stages)):
            q=pi.transpose()*m
            out.extend([q[0,0],q[1,0]])
            if want_deriv:
                dq=pi.transpose()*dm
                for a in (0,1):
                    qder=m.transpose()*ds[a]
                    for j in range(d): jac[2*t+a,j]=dq[a,j]+qder[0,j]
            if t==nt-1:break
            mass_next=self.H.transpose()*m
            if want_deriv: dm_next=self.H.transpose()*dm
            for a in range(3):
                mp=M(S,1)
                if want_deriv:dmp=M(S,d)
                for i in range(a*n,S):
                    mp[i,0]=m[i,0]*pi[i,a]
                    if want_deriv:
                        for j in range(d):
                            dmp[i,j]=pi[i,a]*dm[i,j]+m[i,0]*ds[a][i,j]
                kt=K[a].transpose()
                mass_next+=kt*mp*self.jump
                if want_deriv:
                    dm_next+=kt*dmp*self.jump
                    if a:
                        for l in (0,1):
                            tmp=dK[a][l].transpose()*mp*self.jump
                            for i in range(S):dm_next[i,2*t+l]+=tmp[i,0]
            m=mass_next
            if want_deriv:dm=dm_next
        return out,jac

def run(n,nt,power,precision):
    start=time.monotonic();ctx.prec=precision
    w=json.loads((ROOT/f'results/center_{n}_{nt}.json').read_text())
    center=[[A(k)/w['scale'] for k in row] for row in w['center']]
    flat=[x for pair in center for x in pair];r=ONE/(2**power)
    game=Grid(n,nt)
    y,_=game.sweep(center,False)
    defect=maxupper(abs(a-b) for a,b in zip(y,flat))
    game.stats=dict(interior_events=0,multicell_events=0)
    box=[[x+A(0,r) for x in pair] for pair in center]
    _,J=game.sweep(box,True)
    gain=row_norm(J)
    ok=bool(gain<1 and defect+gain*r<r)
    result=dict(nodes=n,steps=nt,dimension=2*nt,bits=precision,radius_power=power,
        center_defect_upper=upper_decimal(defect,18), jacobian_norm_upper=upper_decimal(gain),
        invariant_and_contracting=ok,**game.stats,
        posterior_endpoints='exact identities for every time and action',
        preconditioner='identity',elapsed_seconds=round(time.monotonic()-start,2))
    if ok:
        dist=defect/(1-gain)
        result['root_coordinate_distance_upper']=upper_decimal(dist,18)
        result['root_tv_distance_upper']=upper_decimal(2*dist,18)
    path=ROOT/f'results/cert_{n}_{nt}_r{power}_p{precision}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('nodes',type=int);p.add_argument('steps',type=int)
    p.add_argument('--power',type=int,default=28);p.add_argument('--bits',type=int,default=128)
    a=p.parse_args();run(a.nodes,a.steps,a.power,a.bits)
