"""Use the archived float64 CalibModel only to propose a decimal witness and cross-check."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'prior'))
from calib_model import CalibModel

def setup(n, steps):
    data = json.loads((ROOT/'data/archived.json').read_text())['criteo_offsets']['primitives']
    return CalibModel(n, steps, .45, .05, 1., 2.25, np.arange(4)/20,
                      data['q0'], data['zeta'], data['d'], data['Delta'], data['p0'])

def paths(method, t):
    data = json.loads((ROOT/f'data/{method}_path.json').read_text())
    q = np.array(data['nodal_flow']) / data['scale']
    return np.column_stack([np.interp(t, np.linspace(0,1,len(q)), a) for a in q.T])

def response(model, q):
    return model.forward(q, model.backward(q)[1])[1]

def run(n, steps):
    model = setup(n, steps)
    t = np.arange(steps)/steps
    initial = paths('fem',t)
    q = initial.copy()
    for k in range(100):
        u = response(model, q)
        if np.max(abs(u-q)) < 5e-15: break
        q = u
    else: raise RuntimeError('iteration limit')
    scale = 10**14
    center = np.rint(q[:,:2]*scale).astype(int)
    q = np.column_stack((center/scale,1-center.sum(axis=1)/scale))
    def max_tv(x):return float(np.max(np.abs(x-q).sum(axis=1)/2))
    record = dict(nodes=n, steps=steps, scale=scale, center=center.tolist(), iterations=k+1,
        fem_sample_max_tv=max_tv(initial), pinn_sample_max_tv=max_tv(paths('pinn',t)),
        original_float_max_residual=max_tv(response(model,q)))
    (ROOT/f'results/center_{n}_{steps}.json').write_text(json.dumps(record,indent=1)+'\n')
    print(n, steps, k+1, record['original_float_max_residual'],record['fem_sample_max_tv'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('nodes',type=int);p.add_argument('steps',type=int)
    a=p.parse_args();run(a.nodes,a.steps)
