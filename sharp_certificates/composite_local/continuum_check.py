"""Test the continuous-time proof's gain on a band around the exact archived FEM witness."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'continuum/proof'))
import budget_gain as bg
from flint import arb as A,ctx

def run(tol,cells):
    ctx.prec=224
    q=json.loads((ROOT/'data/fem_path.json').read_text())
    lo=[max(0,min(a[j]*10**6//q['scale'] for a in q['nodal_flow'])-tol) for j in range(3)]
    hi=[min(10**6,max(-(-a[j]*10**6//q['scale']) for a in q['nodal_flow'])+tol) for j in range(3)]
    # Includes every point of the linear witness and its full TV tube of the requested radius.
    box=(lo,hi);state=bg.model('archived','.45',cells,True,box);K=bg.kernels(state)
    fs=bg.model('archived','.45',cells,False,box)
    fk=[[[([float(v) for v in a],[float(v) for v in b]) for a,b in rr] for rr in R] for R in K]
    w=np.ones(cells)
    for _ in range(45):
        y=np.asarray(bg.evalw(fs,fk,w));w=y/max(y)
    weights=[A(f'{v:.12f}') for v in w];assert all(v>0 for v in weights)
    y=bg.evalw(state,K,weights)
    upper=state.math_ops.maximum(b/a for b,a in zip(y,weights))
    gain_un=state.math_ops.maximum(bg.evalw(state,K,[A(1)]*cells))
    result=dict(tube_radius_millionths=tol,box_in_millionths=box,cells=cells,bits=ctx.prec,
        weighted_gain=str(upper),unweighted_gain=str(gain_un),sufficient_gain_below_one=bool(upper<1),
        weights=[f'{v:.12f}' for v in w],scope='continuous-time continuous-belief comparison on the restricted convex box; no invariance asserted')
    (ROOT/f'results/continuous_tube_{tol}_{cells}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(tol,cells,result['weighted_gain'],result['unweighted_gain'],flush=True)
if __name__=='__main__':run(int(sys.argv[1]),int(sys.argv[2]))
