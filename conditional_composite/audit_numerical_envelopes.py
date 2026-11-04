"""Independent floating-point plausibility audit; not proof evidence."""
import sys,json
from pathlib import Path
import numpy as np
from flint import arb
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'transfer'))
from full_120d_solver import Model
from conditional_gain import bg
r=json.loads((ROOT/'continuous_certificate_240cells.json').read_text())
lo=np.array([[float(arb(x)) for x in row] for row in r['diagnostics']['conditional_lower_t0']])
hi=np.array([[float(arb(x)) for x in row] for row in r['diagnostics']['conditional_upper_t0']])
p=json.loads((ROOT/'transfer/fitted_parameters.json').read_text())
m=Model(np_grid=101,nt=1000,eta=.45,eps=.05,zeta=p['zeta'],base=p['q0'],delta=p['d'],values=p['Delta'],mgrid=p['mgrid'],p0=p['p0'])
vertices=np.array(bg.vertices(*r['box']))/1e6
checks=[]
flows=[('vertex_'+str(i),np.tile(v,(m.nt,1))) for i,v in enumerate(vertices)]
flows.append(('rapid_switch',vertices[np.arange(m.nt)%len(vertices)]))
for label,q in flows:
    V,_=m.backward(q);dv=np.gradient(V[0],m.h,axis=1,edge_order=2)
    alpha=np.array([V[0]-m.p*dv,V[0]+(1-m.p)*dv])
    margin=float(min(np.min(alpha-lo[:,:,None]),np.min(hi[:,:,None]-alpha)))
    nonzero=float(min(np.min((alpha-lo[:,:,None])[:,1:]),np.min((hi[:,:,None]-alpha)[:,1:])))
    checks.append(dict(flow=label,minimum_margin_nonzero_budget=nonzero,all_within=bool(margin>=-1e-10)))
    print(checks[-1],flush=True)
assert all(x['all_within'] for x in checks)
(ROOT/'numerical_envelope_audit.json').write_text(json.dumps({'scope':'floating-point derivative-of-value diagnostic only; not part of the proof','belief_nodes':101,'time_steps':1000,'checks':checks},indent=2)+'\n')
