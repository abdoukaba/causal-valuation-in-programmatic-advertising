"""Reproducible comment-8 analysis on the checksum-verified Criteo v2.1 release.

Uses the original seed-3 two-fold group-learning algorithm. Gaussian multiplier
bootstrap of the joint empirical moments is evaluated through its exact normal
covariance, avoiding a rows-by-replicates allocation. This conditions on nuisance
fits and learned groups: it is NOT a full-pipeline bootstrap or a deployment CI.
"""
import argparse, hashlib, json, os, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
import joblib

SHA='2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc'
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert sha(a.csv)==SHA, 'Wrong release/checksum'
    cache=out/'arrays.npz'
    if not cache.exists():
        df=pd.read_csv(a.csv,dtype='float32')
        np.savez(cache,X=df[[f'f{i}' for i in range(12)]].to_numpy(),
                 Z=df.treatment.to_numpy().astype('int8'),Y=df.visit.to_numpy().astype('int8'),
                 D=df.exposure.to_numpy().astype('int8'))
        del df
    d=np.load(cache);X=d['X'];Z=d['Z'];Y=d['Y'];D=d['D'];n=len(Z)
    assert n==13979592 and not D[Z==0].any()
    rng=np.random.default_rng(3);fold=rng.integers(0,2,n)
    pred=np.zeros((n,4))
    fitmeta=[]
    for k in range(2):
        tr=np.where(fold!=k)[0];te=np.where(fold==k)[0]
        t1=tr[Z[tr]==1];t0=tr[Z[tr]==0]
        for j,(name,idx,y,cap) in enumerate([('z',tr,Z,2000000),('d1',t1,D,2000000),('m1',t1,Y,2000000),('m0',t0,Y,len(t0))]):
            chosen=rng.choice(idx,min(cap,len(idx)),replace=False)
            file=out/f'model_fold{k}_{name}.joblib'
            start=time.time()
            if file.exists(): model=joblib.load(file)
            else:
                model=HGB(max_iter=200,random_state=0).fit(X[chosen],y[chosen])
                joblib.dump(model,file)
            pp=out/f'prediction_fold{k}_{name}.npy'
            if pp.exists(): p=np.load(pp)
            else:
                p=model.predict_proba(X[te])[:,1];np.save(pp,p)
            pred[te,j]=p
            fitmeta.append(dict(fold=k,target=name,n_train=len(chosen),iterations=int(model.n_iter_),
                                model_sha256=sha(file)))
            print('fit',k,name,'seconds',round(time.time()-start,1),flush=True)
        del tr,te,t1,t0
    np.save(out/'nuisances.npy',pred)
    del X,d,fold
    zz,md,m1,m0=pred.T
    score=(m1-m0)/np.maximum(md,.001);cut=float(np.median(score));high=score>cut
    np.save(out/'group_high.npy',high)
    # Distinct clipping runs preserve the original learned groups and all rows.
    result=dict(n=n,sha256=SHA,python=platform.python_version(),sklearn=sklearn.__version__,
       numpy=np.__version__,seed=3,folds=2,clip_default=.02,max_iter=200,training_cap=2000000,
       score_threshold=cut,group_definition='out-of-fold (m1-m0)/max(mD1,0.001) > full-sample median',
       bootstrap=dict(type='joint-moment Gaussian multiplier',draws=a.draws,seed=20261102,
                      nuisance_and_partition_fixed=True,full_pipeline=False,level=.95),fits=fitmeta,runs={})
    for clip in [.01,.02,.05]:
        z=np.clip(zz,clip,1-clip)
        psiY=m1-m0+Z*(Y-m1)/z-(1-Z)*(Y-m0)/(1-z)
        psiD=md+Z*(D-md)/z
        # Rows of joint moments; raw arm denominators are estimated within each group.
        M=np.column_stack([Z*Y,(1-Z)*Y,Z*D,Z,1-Z,
            Z*Y/z-(1-Z)*Y/(1-z),Z*D/z,m1-m0,md,psiY,psiD])
        groups={};drawvals={}
        for label,mask in [('low',~high),('high',high)]:
            v=M[mask];ng=len(v);mu=v.mean(0);cov=np.cov(v,rowvar=False,ddof=1)/ng
            # Exact Gaussian multiplier law of the centered sample moment vector.
            brng=np.random.default_rng(20261102+(label=='high'))
            bm=brng.multivariate_normal(mu,cov,size=a.draws,check_valid='raise')
            estimators={};bs={}
            for method in ['unadjusted','ipw','regression','aipw']:
                def calc(x):
                    if method=='unadjusted':
                        itt=x[...,0]/x[...,3]-x[...,1]/x[...,4];ex=x[...,2]/x[...,3]
                    else:
                        i={'ipw':5,'regression':7,'aipw':9}[method];itt=x[...,i];ex=x[...,i+1]
                    return itt/ex,itt,ex
                theta,itt,ex=calc(mu);bt,bi,be=calc(bm);bs[method]=bt
                estimators[method]=dict(wald=float(theta),itt=float(itt),exposure=float(ex),
                    se=float(bt.std(ddof=1)),wald_ci95=np.quantile(bt,[.025,.975]).tolist(),
                    exposure_ci95=np.quantile(be,[.025,.975]).tolist(),bootstrap_nonpositive_exposure=int((be<=0).sum()))
            groups[label]=dict(n=ng,share=ng/n,treatment_rate=float(mu[3]),
                clipped_fraction=float(((zz[mask]<clip)|(zz[mask]>1-clip)).mean()),estimators=estimators)
            drawvals[label]=bs
            np.savez(out/f'moments_clip{clip}_{label}.npz',mean=mu,covariance_of_mean=cov,n=ng)
            del v,bm
        ratios={}
        for method in drawvals['low']:
            bl,bh=drawvals['low'][method],drawvals['high'][method];ratio=bh/bl
            ratios[method]=dict(estimate=groups['high']['estimators'][method]['wald']/groups['low']['estimators'][method]['wald'],
                ci95=np.quantile(ratio,[.025,.975]).tolist(),se=float(ratio.std(ddof=1)),
                nonpositive_low_effect_draws=int((bl<=0).sum()),finite_draws=int(np.isfinite(ratio).sum()))
        result['runs'][str(clip)]=dict(groups=groups,ratios=ratios)
        print('clip',clip,json.dumps(ratios),flush=True)
        del M
    (out/'criteo_group_review.json').write_text(json.dumps(result,indent=2))
    print('DONE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--out',required=True);p.add_argument('--draws',type=int,default=10000)
    run(p.parse_args())
