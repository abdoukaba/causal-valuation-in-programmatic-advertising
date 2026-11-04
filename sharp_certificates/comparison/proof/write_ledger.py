"""Generate a detached integer aggregation ledger from the current Arb verifier.

The integer replay certifies only aggregation of these upper-bounded constants;
their mathematical validity and enclosure still depend on Appendix C and Arb.
Run: ARB_BITS=224 EPS=.05 python write_ledger.py .45 output.json.gz
"""
import gzip,json,sys,os
from pathlib import Path
from flint import ctx
import budget_gain as b
from replay_integer import bound

def run(eta,out):
    assert os.environ.get('EPS','.05') in ('.05','0.05')
    ctx.prec=224;n=40;scale=10**10
    box=([0,0,0],[1000000]*3)
    for _ in range(2):
        state=b.model('auction',eta,n,True,box);lo,hi=b.restrict(state)
        box=([max(a,c) for a,c in zip(box[0],lo)],[min(a,c) for a,c in zip(box[1],hi)])
    state=b.model('auction',eta,n,True,box);K=b.kernels(state)
    def up(x):
        assert x>=0
        return int((x*scale).upper().ceil().unique_fmpz())
    cells=[dict(prob=[up(x) for x in c.probs],source_span=[up(x) for x in c.forcing_span],
                direct=[[up(x) for x in row.instant_gain] for row in c.row],
                gibbs_slope=[[up(x) for x in row.gibbs_factor] for row in c.row]) for c in state.c]
    kernels=[[[[[up(x) for x in spans],[up(x) for x in env]] for spans,env in rows] for rows in terminal] for terminal in K]
    ledger=dict(share=eta,n=n,scale=scale,bits=224,box=box,cells=cells,kernels=kernels,
                calibration='auction',weight='identically one')
    Path(out).write_bytes(gzip.compress(json.dumps(ledger,separators=(',',':')).encode(),mtime=0))
    num=bound(ledger);assert num<scale
    print(json.dumps(dict(numerator=num,denominator=scale,strictly_below_one=True)))
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
