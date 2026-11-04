"""Independent numerical diagnostics from original quotient-interpolation implementation."""
import json
import numpy as np
import sys
from pathlib import Path
from fractions import Fraction as F
from flint import arb as A,ctx
from candidate import setup,response
from verify_grid import Grid
ROOT=Path(__file__).resolve().parent

def run(n,nt):
    ctx.prec=128
    record=json.loads((ROOT/f'results/center_{n}_{nt}.json').read_text())
    x=np.array(record['center'])/record['scale']
    def phi(x):
        q=np.column_stack((x,1-x.sum(axis=1)))
        return response(model,q)[:,:2].reshape(-1)
    model=setup(n,nt)
    grid=Grid(n,nt)
    y,J=grid.sweep([[A(v)/record['scale'] for v in row] for row in record['center']],True)
    ys=np.array([float(a.mid()) for a in y])
    orig=phi(x)
    matrix=np.array([[float(J[i,j].mid()) for j in range(2*nt)] for i in range(2*nt)])
    def diff(h):
        err=0.
        for j in range(2*nt):
            step=np.zeros_like(x);step.ravel()[j]=h
            c=(phi(x+step)-phi(x-step))/(2*h)
            err=max(err,np.max(abs(c-matrix[:,j])))
        return err
    errors={str(h):float(diff(h)) for h in (2e-6,5e-7)}
    rng=np.random.default_rng(7501);corners=[]
    for radius in (1/512,1/8192):
        z=x+rng.choice([-1,1],size=x.shape)*radius
        yr,_=grid.sweep([[A(float(a)) for a in row] for row in z],False)
        corners.append(float(max(abs(phi(z)-np.array([float(a.mid()) for a in yr])))))
    out=dict(center_output_max_difference=float(max(abs(ys-orig))),all_jacobian_columns=2*nt,
        fd_max_entry_error_by_step=errors,two_off_center_max_differences=corners,
        jacobian_infinity_norm_at_center_float=float(abs(matrix).sum(axis=1).max()),
        check='finite differences and original quotient-interpolation implementation; diagnostic only')
    assert max(errors.values())<2e-8 and max([out['center_output_max_difference']]+corners)<2e-13
    (ROOT/'results/independent_grid_check.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out,flush=True)
if __name__=='__main__':run(21,60)
