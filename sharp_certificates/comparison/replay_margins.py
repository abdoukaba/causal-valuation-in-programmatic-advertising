"""Recover rigorous precise sweep margins and replay every retained eta ball.

Default: re-evaluate balls in descending order of their archived compact upper
enclosure until all unevaluated balls have smaller proved upper bounds. This
determines the maximum upper bound without recomputing dominated balls.
--all additionally replays every retained ball. Each new call uses the original
float endpoint-to-ball construction, padded radius, fixed cells and precision.
"""
import argparse,json,os,subprocess,sys
from decimal import Decimal,ROUND_CEILING
from pathlib import Path
from flint import arb,ctx
ROOT=Path(__file__).resolve().parent
def up(x):
    n=int((x*10**12).upper().ceil().unique_fmpz());return f'{n//10**12}.{n%10**12:012d}'
def main():
    cli=argparse.ArgumentParser();cli.add_argument('--all',action='store_true');a=cli.parse_args();ctx.prec=224
    old=json.loads((ROOT/'results/uniqueness_sweep_summary.json').read_text())
    ext=json.loads((ROOT/'results_ext/extension_summary.json').read_text())
    out=ROOT/'results_margins';out.mkdir(exist_ok=True);summary=[]
    for row in old:
        m,e=row['model'],row['eps'];items=list(row['certified_intervals']);endpoint=row['certified_hi']
        ex=next((x for x in ext if x['model']==m and Decimal(x['eps'])==Decimal(e)),None)
        if ex: items+=ex['certified_intervals'];endpoint=ex['certified_hi']
        records=[dict(index=i,lo=x['lo'],hi=x['hi'],archive_bound=x['bound'],
                      upper=up(arb(x['bound'])),source='archived enclosing display') for i,x in enumerate(items)]
        # Explicit no-gap coverage check, including extension overlap.
        reach=0.
        for r in sorted(records,key=lambda x:(x['lo'],x['hi'])):
            assert r['lo']<=reach+1e-12,(m,e,reach,r)
            reach=max(reach,r['hi'])
        assert reach>=endpoint-1e-12
        maximum=Decimal(0);fresh=0
        for r in sorted(records,key=lambda x:Decimal(x['upper']),reverse=True):
            if not a.all and Decimal(r['upper'])<=maximum: continue
            f=out/f'{m}_{e}_{r["index"]:04d}.json'
            if f.exists(): d=json.loads(f.read_text())
            else:
                c=(r['lo']+r['hi'])/2;rad=(r['hi']-r['lo'])/2*1.000001
                ball=f'[{c!r} +/- {rad!r}]'
                cp=subprocess.run([sys.executable,str(ROOT/'proof/budget_gain.py'),m,ball,'40','1'],
                    env=dict(os.environ,EPS=e,ARB_BITS='224'),capture_output=True,text=True,check=True)
                d=json.loads(cp.stdout);f.write_text(json.dumps(d,indent=2))
            assert d['below_one'] and Decimal(d['bound_upper'])<1,(m,e,r,d['bound_upper'])
            r.update(upper=d['bound_upper'],source=f.name)
            maximum=max(maximum,Decimal(r['upper']));fresh+=1
            print(m,e,'ball',r['index'],'upper',r['upper'],flush=True)
        maximum=max(Decimal(r['upper']) for r in records)
        assert maximum<1
        rec=dict(model=m,eps=e,certified_hi=endpoint,balls=len(records),max_upper=str(maximum),
          margin_lower=str(1-maximum),fresh_balls=fresh,all_replayed=a.all,cells=40,bits=224,records=records)
        summary.append(rec);(out/'margin_summary.json').write_text(json.dumps(summary,indent=2))
        print('ROW',m,e,'max',maximum,'balls',len(records),'fresh',fresh,flush=True)
if __name__=='__main__':main()
