"""Exact rational comparisons, including both traces at time discontinuities."""
import json
from pathlib import Path
from fractions import Fraction as F
from flint import arb as A,ctx
from verify_grid import upper_decimal
ROOT=Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT/name).read_text())

def linear(q,t):
    n=len(q)-1;k=min(int(t*n),n-1);u=t*n-k
    return tuple(x+(y-x)*u for x,y in zip(q[k],q[k+1]))

def tv(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))/2

def outward(x):
    scale=10**17;v=-(-x.numerator*scale//x.denominator)
    return f'{v//scale}.{v%scale:017d}'

def path(name):
    w=read(f'data/{name}_path.json')
    return [tuple(F(x,w['scale']) for x in row) for row in w['nodal_flow']]

def exact_gap(step,line):
    n=len(step);m=len(line)-1;maxgap=F(0)
    for i,v in enumerate(step):
        nodes=[F(i,n),F(i+1,n)]+[F(k,m) for k in range(i*m//n+1,(i+1)*m//n+1) if F(k,m)<F(i+1,n)]
        maxgap=max(maxgap,max(tv(v,linear(line,t)) for t in nodes))
    return maxgap

def validate_box(rows,box,margin=F(0)):
    for row in rows:
        assert sum(row)==1
        for x,lo,hi in zip(row,*box):
            assert F(lo,10**6)<=x-margin and x+margin<=F(hi,10**6)

def run():
    ctx.prec=224
    witness=read('data/witness_fem_composite_0.45.json')
    step=[tuple(F(x,witness['scale']) for x in row) for row in witness['midpoint_flow']]
    cont=read('results/continuous_dual_160_160_p160.json')
    tight=read('results/continuous_tube_1000_40.json')
    wide=read('results/continuous_tube_10000_40.json')
    exact=A(cont['exact_response_residual_TV_upper'])
    evidence={}
    def put(key,line,margin,band):
        box=band['box_in_millionths'];validate_box(step,box);validate_box(line,box,margin)
        gap=exact_gap(step,line)+margin
        rigorous=exact+(1+A(band['unweighted_gain']))*A(gap.numerator)/gap.denominator
        evidence[key]=dict(distance_to_FEM_step_TV_upper=outward(gap),
            unweighted_continuous_gain=band['unweighted_gain'],extended_flow_continuous_residual_TV_upper=upper_decimal(rigorous))
    fem=path('fem');pinn=path('pinn')
    put('linear_FEM',fem,F(0),tight);put('linear_PINN',pinn,F(0),tight)
    for n,nt,bits in [(11,30,128),(21,60,128),(31,120,160)]:
        case=read(f'results/center_{n}_{nt}.json')
        cert=read(f'results/cert_{n}_{nt}_r28_p{bits}.json')
        assert cert['invariant_and_contracting']
        vals=[tuple(F(v,case['scale']) for v in row) for row in case['center']]
        rows=[(x,y,1-x-y) for x,y in vals]
        # At time one use the last grid ordinate (constant final interval).
        key=f'FD_{n}_{nt}';put(key,rows+[rows[-1]],F(cert['root_tv_distance_upper']),wide)
        for mode,line in [('FEM',fem),('PINN',pinn)]:
            diffs=[tuple(a-b for a,b in zip(linear(line,F(i,nt)),r)) for i,r in enumerate(rows)]
            evidence[key][f'{mode}_sample_coordinates_max_upper']=outward(max(max(abs(x) for x in d[:2]) for d in diffs))
            evidence[key][f'{mode}_sample_TV_max_upper']=outward(max(sum(map(abs,d))/2 for d in diffs))
            if nt in (60,120):
                power={60:9,120:10}[nt]
                wide_cert=read(f'results/cert_{n}_{nt}_r{power}_p128.json')
                assert wide_cert['invariant_and_contracting']
                assert max(max(abs(x) for x in d[:2]) for d in diffs)<F(1,2**power)
    (ROOT/'results/transfer.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=1))
if __name__=='__main__':run()
