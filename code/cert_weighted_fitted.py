import json, cert_general as cg
S=json.load(open('calib_season1.json')); out={}
for top in (450,600):
    res=S[f'top={top}']; q0=res['q0']; mg=[0,150/3000,300/3000,top/3000]; z=str(round(res['zeta_fit'],4))
    qs=[str(round(x,6)) for x in q0[:-1]]; qs.append(str(round(1-sum(round(x,6) for x in q0[:-1]),6)))
    for eta in ('0.3','0.45'):
        best=None
        for al in ('0','1','2'):
            for th in ('4','8','16','32'):
                k=cg.kappa([round(x,6) for x in mg],qs,z,1,'2.25','0.05',eta,th,nt=2000,alpha=al)['kappa'].upper()
                if best is None or k<best[0]: best=(k,th,al)
        out[f'{top},{eta}']=(best[0].str(3,radius=False),best[1],best[2]); print(top,eta,out[f'{top},{eta}'],flush=True)
json.dump(out,open('fitted_weighted.json','w'),indent=1)
