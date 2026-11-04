"""Terminal-opportunity information dividend, outward rounded over every rival simplex point."""
from pathlib import Path
import json
from flint import arb,ctx
ctx.prec=160
root=Path(__file__).resolve().parent
p=json.loads((root/'v5/code/sharp_certificates/comparison/inputs/archived.json').read_text(),parse_float=str)['criteo_offsets']['primitives']
A=lambda x:arb(str(x))
p0=A(p['p0']); eps=A('.05'); eta=A('.45')
a=list(map(A,['0','.05','.10','.15']));q0=list(map(A,p['q0']))
g=[]
for s in range(2):
    row=[[arb(0)]*4]
    for bid in a[1:3]:
        row.append([(A(x)-bid)/(1+(-(bid-m-A(p['d'][s]))/A(p['zeta'])).exp()) for x,m in zip(p['Delta'][s],a)])
    g.append(row)
def lse(scores):
    return eps*sum((x/eps).exp() for x in scores).log()
N=80
def vertex(i,j):
    end=[arb(i)/N,arb(j)/N,arb(N-i-j)/N,arb(0)]
    law=[(1-eta)*x+eta*y for x,y in zip(q0,end)]
    scores=[[sum(x*y for x,y in zip(law,k)) for k in row] for row in g]
    return scores+[[(1-p0)*x+p0*y for x,y in zip(*scores)]]
grid={(i,j):vertex(i,j) for i in range(N+1) for j in range(N+1-i)}
def check(pts):
    rows=[]
    for s in range(3):
        scores=[]
        for k in range(3):
            x=grid[pts[0]][s][k]
            for pt in pts[1:]: x=x.union(grid[pt][s][k])
            scores.append(x)
        rows.append(lse(scores))
    return (1-p0)*rows[0]+p0*rows[1]-rows[2]
gaps=[]
for i in range(N):
    for j in range(N-i):
        gaps.append(check([(i,j),(i+1,j),(i,j+1)]))
        if i+j+1<N:gaps.append(check([(i+1,j),(i,j+1),(i+1,j+1)]))
assert len(gaps)==N*N
scale=10**7
lo=min(int((x*scale).lower().floor().unique_fmpz()) for x in gaps)
hi=max(int((x*scale).upper().ceil().unique_fmpz()) for x in gaps)
assert lo>0
out={'triangle_count':len(gaps),'lower':lo,'upper':hi,'scale':scale,'bits':ctx.prec}
(root/'results/disclosure_bound.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
