"""Continuous-time significance test. Only Arb comparisons confer certification."""
import json, os, numpy as np
from pathlib import Path
from flint import arb,ctx
from itertools import combinations
ctx.prec=int(os.environ.get('ARB_BITS', '140'))
ROOT=Path(__file__).resolve().parents[1]
A=lambda x:arb(str(x))
def mx(values):
    vals=iter(values);z=next(vals)
    for v in vals:z=z.max(v)
    return z
def mn(values):return -mx(-x for x in values)
def dot(a,b):return sum((x*y for x,y in zip(a,b)),A(0))
def stored(which):
    f=json.loads((ROOT/'inputs'/'auction_fit.json').read_text(),parse_float=str)['top=450']
    grid=list(map(A,['0','.05','.1','.15']))
    if which=='tail4h': grid[-1]=A('.20')
    if which in ('auction','auction_rounded','auction_raw','benchmark','benchmark_eps'):
        d=[A('-.01'),A('.01')]
        D=[[A(c)+A('.03')*v for v in grid] for c in ['.06','.095']]
    else:
        x=json.loads((ROOT/'inputs'/(which+'.json')).read_text(),parse_float=str)['criteo_offsets']['primitives']
        d=list(map(A,x['d']));D=[list(map(A,row)) for row in x['Delta']]
        assert x['q0']==f['q0'] and x['zeta']==f['zeta_fit']
    q=list(map(A,f['q0']))
    if which in ('benchmark','benchmark_eps'):   # the paper's certified benchmark (Table 2): q0=(.2,.4,.4) on {0,h,2h}, zeta=.2
        d=[A('-.01'),A('.01')]; D=[[A(c)+A('.03')*v for v in grid] for c in ['.06','.095']]
        return grid,[A('.2'),A('.4'),A('.4'),A(0)],A('.2'),d,D
    if which=='auction_raw':
        raw=json.loads((ROOT/'inputs'/'raw_counts.json').read_text())
        q=[A(v)/raw['joined_rows'] for v in raw['counts']]
    return grid,q,A('.011' if which=='auction_rounded' else f['zeta_fit']),d,D
def constants(which,eta):
    M,q,z,d,D=stored(which); eta=A(eta)
    V=[[(1-eta)*x+eta*int(k==j) for j,x in enumerate(q)] for k in range(3)]
    out=[];ratios=[];extrema=[]
    for a in M[1:3]:
        W=[[1/(1+(-(a-m-off)/z).exp()) for m in M]for off in d]
        G=[[w*(v-a)for w,v in zip(ws,vs)] for ws,vs in zip(W,D)]
        ell=[[dot(v,ws)for ws in W]for v in V]
        rew=[[dot(v,gs)for gs in G]for v in V]
        extrema.extend(y for row in ell for y in row)
        for j in range(4):
            w0,w1=W[0][j],W[1][j]
            ratios.extend([w0/w1,w1/w0,(1-w0)/(1-w1),(1-w1)/(1-w0)])
        O=mx(abs(W[s][i]-W[s][j])for s in range(2) for i,j in combinations(range(3),2))
        Gosc=mx(abs(G[s][i]-G[s][j])for s in range(2) for i,j in combinations(range(3),2))
        Lw=mx(abs(y-x)for x,y in ell);Lg=mx(abs(y-x)for x,y in rew)
        beta=A(0)
        for i,j in combinations(range(3),2):
            h0=W[0][i]-W[0][j];h1=W[1][i]-W[1][j]
            b=A(0)
            for rows in (ell,[tuple(1-x for x in r)for r in ell]):
                nom=mx(abs(x*h1-y*h0)for x,y in rows)
                den=(mn(x for x,y in rows).sqrt()+mn(y for x,y in rows).sqrt())**2
                b+=nom/den
            beta=beta.max(b)
        spread=A(0)
        # Posterior separation via upper/lower odds ratios over the convex hull.
        for s in [0,1]:
            ratio=mx(v[s]/v[1-s] for v in ell)*mx((1-v[1-s])/(1-v[s]) for v in ell)
            sroot=ratio.sqrt();spread=spread.max((sroot-1)/(sroot+1))
        out.append((O,Gosc,Lw,Lg,beta,spread,mx(y for r in rew for y in r)))
    return out,(mn(extrema),mx(extrema),mx(ratios))
def arrays(which,eta,n):
    e=A(os.environ.get('EPS','.05'));lam=A('2.25');h=A(1)/n;pc,diag=constants(which,eta);eta=A(eta)
    # Maximum attainable one-stage Shannon value using the largest expected rewards for each bid.
    v=lam*e*(1+sum((row[6]/e).exp() for row in pc)).log()
    lw=mx(a[2]for a in pc);lg=mx(a[3]for a in pc)
    D=[];H=[]
    for j in range(n):
        s=(n-j)*h;B=v*s;L=lam*(lg*s+lw*v*s*s/2)
        D.append(eta/2*mx(g+o*(B+ls*L)+be*L for o,g,_,_,be,ls,_ in pc))
        H.append((L+mx(lg0+lw0*B for _,_,lw0,lg0,_,_,_ in pc))/e)
    return pc,diag,D,H
def operator(w,pc,D,H,eta):
    n=len(D);dt=A(1)/n;l=A('2.25');e=A(os.environ.get('EPS','.05'));eta=A(eta)
    E=A(0);P=[None]*n
    for k in reversed(range(n)):
        E+=l*dt*D[k]*w[k]
        P[k]=(D[k]*w[k]+E)/e
    om=mx(r[0]for r in pc);bw=mx(r[4]for r in pc);lw=mx(r[2]for r in pc)
    # Terminal test f=pi dot sign; use exact integral of additive Lipschitz growth, no exponent.
    a=b=c=A(0);out=[]
    for k in range(n):
        b+=dt*w[k];c+=l*dt*(2*lw+H[k])*b
        a+=dt*(P[k]+eta*om*w[k])
        out.append(P[k]+l*(a+eta*bw/2*(b*H[k]+c)))
    return out
def floatweights(pc,D,H,eta):
    n=len(D);om=float(mx(r[0]for r in pc));bw=float(mx(r[4]for r in pc));lw=float(mx(r[2]for r in pc))
    D=np.array(list(map(float,D)));H=np.array(list(map(float,H)));w=np.ones(n);la=2.25
    for it in range(100):
        dw=D*w
        p=(dw+la/n*np.cumsum(dw[::-1])[::-1])/.05
        b=np.cumsum(w)/n;c=la/n*np.cumsum((2*lw+H)*b)
        y=p+la*(np.cumsum(p+eta*om*w)/n+eta*bw/2*(b*H+c))
        w=y/y.max()
    return [A(f'{v:.12f}') for v in w]
def evaluate(which,eta,n,full=False):
    pc,diag,D,H=arrays(which,eta,n); w=floatweights(pc,D,H,float(eta))
    out=operator(w,pc,D,H,eta)
    k=mx(x/y for x,y in zip(out,w));d=dict(calibration=which,eta=str(eta),cells=n,bound=k.str(14),certified=bool(k<1),ell_min=diag[0].str(10),ell_max=diag[1].str(10),old_ratio=diag[2].str(10))
    if full:d.update(primitive_bounds=[[v.str(12)for v in r]for r in pc],D0=D[0].str(10),H0=H[0].str(10),w=[round(float(x),12) for x in w])
    return d
if __name__=='__main__':
    import sys
    for model in ['auction','archived','refit']:
        for eta in ['.05','.1','.2','.3','.45']:
            r=evaluate(model,eta,int(sys.argv[1])if len(sys.argv)>1 else 100);print(r,flush=True)
