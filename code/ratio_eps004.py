import json, grid_certificate as gc
out={}
for eta in (.45,.5):
    for eps in (0.04,0.03):
        m=gc.Model(21,60,eta,eps,1.,2.25); sol=m.solve(0,max_iter=800,damping=.5); q=sol['q']
        mon=m.monetary_policy_value(q,sol['pol']); br=m.monetary_policy_value(q,None)
        out[f'{eta},{eps}']=dict(mon=mon,br=br,ratio=br/mon,res=sol['res'],steps=sol['steps']); print(eta,eps,out[f'{eta},{eps}'],flush=True)
json.dump(out,open('ratio_eps004.json','w'),indent=1)
