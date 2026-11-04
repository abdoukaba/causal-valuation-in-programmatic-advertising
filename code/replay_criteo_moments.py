"""Replay the conditional bootstrap from saved joint moments, without raw data."""
import json,sys
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]);saved=json.loads((root/'criteo_group_review.json').read_text())
out={};worst=0.
for clip,run in saved['runs'].items():
    draws={};point={}
    for lab in ['low','high']:
        d=np.load(root/f'moments_clip{clip}_{lab}.npz');mu=d['mean'];cov=d['covariance_of_mean']
        x=np.random.default_rng(20261102+(lab=='high')).multivariate_normal(mu,cov,size=10000,check_valid='raise')
        def effects(v):
            raw=(v[...,0]/v[...,3]-v[...,1]/v[...,4])/(v[...,2]/v[...,3])
            return dict(unadjusted=raw,ipw=v[...,5]/v[...,6],regression=v[...,7]/v[...,8],aipw=v[...,9]/v[...,10])
        draws[lab]=effects(x);point[lab]=effects(mu)
    result={}
    for method in point['low']:
        q=draws['high'][method]/draws['low'][method]
        ci=np.quantile(q,[.025,.975]);est=float(point['high'][method]/point['low'][method])
        ref=run['ratios'][method]
        err=max(abs(est-ref['estimate']),float(np.max(abs(ci-np.array(ref['ci95'])))))
        worst=max(worst,err);assert err<1e-10,(clip,method,err)
        result[method]=dict(estimate=est,ci95=ci.tolist())
    out[clip]=result
print(json.dumps(dict(maximum_discrepancy=worst,ratios=out),indent=2))
