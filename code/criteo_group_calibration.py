"""Ratio-CI endpoint and joint estimator-input calibration sensitivity.

Run from a directory containing the archived results required by calib_full.
The base model's iPinYou law, score scale, monetary normalization and (in the
ratio-only runs) group exposure rates/prior remain fixed. No new uniqueness
certificate is inferred. Every equilibrium uses three starts, 21x60 grid.
"""
import argparse,json
from pathlib import Path
import numpy as np
import calib_full as cf

def run(input_file,output):
    d=json.loads(Path(input_file).read_text());r=d['runs']['0.02'];out={}
    def evaluate(P):
        profits=cf.single_auction_profit(P);a=int(np.argmax(profits))
        rho=np.full(3,.05);rho[a]=.9
        es=cf.equilibrium(P,None,seeds=3);ea=cf.equilibrium(P,rho,seeds=3)
        return dict(selected_bid_index=a,single_auction_profit=profits.tolist(),
            shannon=es,anchored=ea,selection=cf.selection_split(P),
            primitives={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in P.items()})
    for method in ['unadjusted','ipw','regression','aipw']:
        l=r['groups']['low']['estimators'][method];h=r['groups']['high']['estimators'][method]
        for name,ratio in [('point',r['ratios'][method]['estimate']),('lower',r['ratios'][method]['ci95'][0]),('upper',r['ratios'][method]['ci95'][1])]:
            # Keep sign of the actual group effects, including regression-only.
            w=(l['wald'],l['wald']*ratio)
            assert (1-cf.P0)*w[0]+cf.P0*w[1]>0
            tag=f'{method}_ratio_{name}';out[tag]=evaluate(cf.primitives(wald=w))
            out[tag]['ratio']=ratio;out[tag]['type']='ratio_only'
            print(tag,json.dumps({k:out[tag][k] for k in ['ratio','selected_bid_index']}),flush=True)
            Path(output).write_text(json.dumps(out,indent=2))
        tag=f'{method}_joint_inputs'
        out[tag]=evaluate(cf.primitives(wald=(l['wald'],h['wald']),expo=(l['exposure'],h['exposure']),p0=r['groups']['high']['share']))
        out[tag]['type']='joint_estimator_point';print(tag,flush=True)
        Path(output).write_text(json.dumps(out,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args();run(a.input,a.output)
