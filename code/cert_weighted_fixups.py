import json, grid_certificate as gc, cert_general as cg, cert_calib as cc, cert_interval as ci
out={'ratios':{}, 'censored':{}, 'edges':{}}
# 1. best-response / equilibrium ratios at every verified eps (eta=0.45), saved
for eps in (1, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04):
    m=gc.Model(21,60,.45,eps,1.,2.25); sol=m.solve(0,max_iter=800,damping=.5); q=sol['q']
    mon=m.monetary_policy_value(q,sol['pol']); br=m.monetary_policy_value(q,None)
    out['ratios'][str(eps)]=dict(mon=mon,br=br,ratio=br/mon,res=sol['res']); print('ratio',eps,round(br/mon,3),flush=True)
# 2. weighted bound at the floor-censored iPinYou law
D=json.load(open('floor_censored.json')); qc=D['q0_censored']
qs=[str(round(x,6)) for x in qc[:-1]]; qs.append(str(round(1-sum(round(x,6) for x in qc[:-1]),6)))
for zeta in ('0.1','0.2'):
    for eta in ('0.3','0.45'):
        b=None
        for al in ('0','0.5','1','1.5','2','3'):
            for th in ('4','8','10','12','14','16','18','20','25','30'):
                k=float(cg.kappa([0,.05,.1,.15],qs,zeta,1,'2.25','0.05',eta,th,nt=2000,alpha=al)['kappa'].upper())
                if b is None or k<b[0]: b=(k,th,al)
        out['censored'][f'{zeta},{eta}']=b; print('censored',zeta,eta,b,flush=True)
# 3. grid-edge cases with an extended grid
TH=('10','16','20','25','30','40','50','60','80'); AL=('1','2','3','4','5','6','8')
import calib_full as cf
P=cf.primitives(); Ds=[[repr(float(x)) for x in r] for r in P['Delta']]; ds=[repr(float(x)) for x in P['d']]; qq=[repr(float(x)) for x in P['q0']]
out['edges']['calibrated_zeta0.2']=cc.best(cf.MG.tolist(),qq,Ds,ds,'0.2',1,'2.25','0.05','0.45',thetas=TH,alphas=AL); print('edge calib',out['edges']['calibrated_zeta0.2'],flush=True)
b=None
for al in AL:
    for th in TH:
        k=float(cg.kappa([0,.05,.1],['0.2','0.4','0.4'],'0.05',1,'2.25','0.05','0.45',th,nt=2000,alpha=al)['kappa'].upper())
        if b is None or k<b[0]: b=(k,th,al)
out['edges']['robust_zeta0.05']=b; print('edge robust',b,flush=True)
json.dump(out,open('fixups.json','w'),indent=1)
