"""A posteriori gain profile using the existing positive continuous-time comparison operator."""
import json,sys
from decimal import Decimal
from pathlib import Path
import budget_gain as b
from flint import arb as A,ctx
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
def up(a):
    return (int((a*10**10).upper().ceil().unique_fmpz())+1)/10**10
def run(eta,case):
    ctx.prec=224
    g=json.loads((OUT/f'gain_{case}_{eta}.json').read_text());mesh=40
    if case=='t9':
        from t9_gain import replace_inputs
        replace_inputs()
    state=b.model('auction',eta,mesh,True,g['outer_boxes'][-1]);K=b.kernels(state);op=lambda x:b.evalw(state,K,x)
    unit=state.math_ops.maximum(op([A(1)]*mesh));assert unit<1
    for method in ['fem','pinn']:
        f=OUT/f'certificate_{method}_{case}_0{eta}.json';d=json.loads(f.read_text(),parse_float=Decimal);e=list(map(lambda v:A(str(v)),d['residual_cells']))
        cells=len(e)//mesh;assert len(e)==cells*mesh
        E=[state.math_ops.maximum(e[j*cells:(j+1)*cells]) for j in range(mesh)]
        ds=[state.math_ops.maximum(E)/(1-unit)]*mesh;history=[]
        for _ in range(18):
            new=[a+b1 for a,b1 in zip(E,op(ds))]
            ds=[A(str(up(a.min(s)))) for a,s in zip(new,ds)]
            history.append(up(state.math_ops.maximum(ds)))
        result=dict(method=method,eta=eta,unweighted_gain=unit.str(16),iterations=len(history),cell_bounds=[up(x) for x in ds],
                    upper_step_flow=history[-1],upper_published_linear_flow=up(state.math_ops.maximum(ds)+A(str(d['displayed_flow_transfer']))),
                    bounds_after_iterations=history)
        (OUT/f'radius_{method}_{case}_0{eta}.json').write_text(json.dumps(result,indent=2));print(result['upper_published_linear_flow'],flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'ipinyou')
