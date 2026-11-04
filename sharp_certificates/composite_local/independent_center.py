"""Detached interval check of the 120-coordinate center defect via Bayesian quotients.
No Arb, first-party solver, NumPy or derivative code is imported.
"""
import json
from pathlib import Path
from mpmath import iv
ROOT=Path(__file__).resolve().parent

def check():
    iv.dps=75;A=iv.mpf
    w=json.loads((ROOT/'results/center_21_60.json').read_text())
    d=json.loads((ROOT/'data/archived.json').read_text(),parse_float=str)['criteo_offsets']['primitives']
    n,nt=w['nodes'],w['steps'];eta=A('0.45');eps=A('0.05');lam=A('2.25')
    p=[A(i)/(n-1) for i in range(n)]
    u=[(A('.2')*b*(1-b)*(n-1))**2/2 for b in p]
    s0=A(d['p0']);k=int(float((s0*(n-1)).mid))
    assert k<=s0*(n-1) and s0*(n-1)<k+1
    base=list(map(A,d['q0']))
    W=[[[1/(1+iv.exp((A(j-a)/20+A(d['d'][s]))/A(d['zeta']))) for j in range(4)] for a in range(3)] for s in (0,1)]
    def hat(i,joint,prob):
        assert prob>0
        if i==0 or i==n-1:return [(i,A(1))]
        r=joint/prob*(n-1);k=int(float(r.mid))
        assert k<r and r<k+1
        return [(k,k+1-r),(k+1,r-k)]
    def zero():return [[A(0) for i in range(n)] for b in range(5)]
    v=zero();stages=[]
    for t in reversed(range(nt)):
        x=[A(v)/w['scale'] for v in w['center'][t]];x.append(1-sum(x))
        q=[(1-eta)*b+(eta*x[j] if j<3 else 0) for j,b in enumerate(base)]
        auc={}
        for a in (1,2):
            ell=[sum(q[j]*W[s][a][j] for j in range(4)) for s in (0,1)]
            g=[sum(q[j]*W[s][a][j]*(A(d['Delta'][s][j])-A(a)/20) for j in range(4)) for s in (0,1)]
            for i,bp in enumerate(p):
                prob=(1-bp)*ell[0]+bp*ell[1]
                auc[a,i]=(prob,hat(i,bp*ell[1],prob),hat(i,bp*(1-ell[1]),1-prob),(1-bp)*g[0]+bp*g[1])
        new=zero();policy={}
        for b in range(5):
            for i in range(n):
                weights=[A(1)]
                for a in range(1,min(b,2)+1):
                    prob,win,loss,reward=auc[a,i]
                    sc=reward+prob*sum(c*v[b-a][j] for j,c in win)+(1-prob)*sum(c*v[b][j] for j,c in loss)
                    weights.append(iv.exp((sc-v[b][i])/eps))
                den=sum(weights);policy[b,i]=[z/den for z in weights]
                diff=(v[b][min(n-1,i+1)]+v[b][max(0,i-1)]-2*v[b][i])*u[i]
                new[b][i]=v[b][i]+(diff+lam*eps*iv.ln(den))/nt
        v=new;stages.append((policy,auc))
    m=zero();m[4][k]=k+1-s0*(n-1);m[4][k+1]=s0*(n-1)-k
    worst=A(0)
    for t,(policy,auc) in enumerate(reversed(stages)):
        qout=[A(0),A(0),A(0)];nxt=zero()
        for b in range(5):
            for i in range(n):
                mi=m[b][i];dr=mi*u[i]/nt
                nxt[b][i]+=(1-lam/nt)*mi-2*dr
                nxt[b][min(i+1,n-1)]+=dr;nxt[b][max(0,i-1)]+=dr
                for a,h in enumerate(policy[b,i]):
                    qout[a]+=h*mi;jump=lam*mi*h/nt
                    if a==0:nxt[b][i]+=jump
                    else:
                        prob,win,lose,_=auc[a,i]
                        for j,c in win:nxt[b-a][j]+=c*prob*jump
                        for j,c in lose:nxt[b][j]+=c*(1-prob)*jump
        for a in (0,1):
            diff=abs(qout[a]-A(w['center'][t][a])/w['scale'])
            if diff.b>worst.b:worst=diff
        m=nxt
    assert worst.b<A('0.000000000000005151')
    r=dict(nodes=n,steps=nt,decimal_digits=iv.dps,max_coordinate_defect_interval=str(worst),
           verified_threshold='0.000000000000005151',passed=True,scope='center defect only')
    (ROOT/'results/mpmath_center.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r,flush=True)
if __name__=='__main__':check()
