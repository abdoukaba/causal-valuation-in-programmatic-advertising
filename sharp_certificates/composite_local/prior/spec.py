"""Read official archived decimals without converting inputs through binary floating point."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def primitives(case,number):
    raw=json.loads((ROOT/'inputs/auction_fit.json').read_text(),parse_float=str)['top=450']
    grid=[number(i)/20 for i in range(4)]
    q0=list(map(number,raw['q0']));z=number(raw['zeta_fit']);p0=number('.5')
    ds=list(map(number,['-.01','.01']));values=[[number(v)+number('.03')*j for j in grid] for v in ['.060','.095']]
    if case=='composite':
        h=json.loads((ROOT/'inputs/archived.json').read_text(),parse_float=str)['criteo_offsets']['primitives']
        ds=list(map(number,h['d']));values=[list(map(number,t)) for t in h['Delta']];p0=number(h['p0'])
    elif case=='t9':
        r=json.loads((ROOT/'inputs/t9_fit.json').read_text())
        q0=[number(c)/number(r['train_n']) for c in r['counts']];z=number(r['zeta'])
    elif case!='ipinyou':raise ValueError(case)
    return grid,q0,z,ds,values,p0
