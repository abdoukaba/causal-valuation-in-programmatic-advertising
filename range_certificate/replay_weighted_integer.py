"""Detached gain aggregation. Python integers only; no numerical libraries.

Certifies the positive-operator aggregation given upper-bounded input constants.
It does not independently establish the analytic inequalities or Arb enclosures.
"""
import json,gzip,sys
from fractions import Fraction
def ceildiv(a,b):return -(-a//b)
def bound(L):
    epsilon=Fraction(L['epsilon'])
    assert epsilon > 0
    n,S=L['n'],L['scale'];cells=L['cells'];weights=L['weights']
    assert L['weight_scale']==S and len(cells)==len(weights)==n and min(weights)>0
    def mul(a,b):return ceildiv(a*b,S)
    def integral(a):return ceildiv(9*a,4*n)
    def occ(vals,prob):return ceildiv(sum(p*max(vals[4-min(4,k):]) for k,p in enumerate(prob)),S)
    future=[0]*5;policies=[None]*n
    for i in reversed(range(n)):
        c=cells[i]
        future=[f+integral(mul(h,weights[i])) for f,h in zip(future,c['source_span'])]
        policies[i]=[[mul(d,weights[i])+ceildiv(epsilon.denominator*mul(h,future[b]),epsilon.numerator) for d,h in zip(c['direct'][b],c['gibbs_slope'][b])] for b in range(5)]
    ratios=[]
    for t,terminal in enumerate(L['kernels']):
        vals=[]
        for j,rows in enumerate(terminal):
            val=occ([p[j] if j<len(p) else 0 for p in policies[t]],cells[t]['prob'])
            total=0
            for i,(spans,environments) in enumerate(rows):
                parts=[mul(sp,max(policies[i][b]))+mul(env,weights[i]) for b,(sp,env) in enumerate(zip(spans,environments))]
                total+=occ(parts,cells[i]['prob'])
            vals.append(val+integral(total))
        ratios.append(ceildiv(max(vals)*S,weights[t]))
    return max(ratios)
if __name__=='__main__':
    L=json.loads(gzip.decompress(open(sys.argv[1],'rb').read()));r=bound(L)
    print(json.dumps({'numerator':r,'denominator':L['scale'],'bound':f"{r//L['scale']}.{r%L['scale']:012d}",'strictly_below_one':r<L['scale'],'scope':'aggregation of supplied upper-bound ledger only'},indent=2))
