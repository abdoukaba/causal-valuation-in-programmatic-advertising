#!/usr/bin/env python3
"""Small deterministic arithmetic/geometry consistency checks; not a substitute for the theorem."""
import json, random
from fractions import Fraction
from pathlib import Path
from independent_interval_replay import (I,S,Z,U,lo,hi,box_minmax,simplex_vertices,
    exact_bound,ceil_scaled,floor_scaled,Replay,contains_zero,certain_lt)

def rational_contains(ball,value):
    a,b=exact_bound(ball,'lower'),exact_bound(ball,'upper')
    return Fraction(*a)<=value<=Fraction(*b)

def check_lp():
    random.seed(20261116)
    successes=0
    for d in (2,3):
        for _ in range(80):
            l=[random.randrange(0,3) for _ in range(d)]
            h=[random.randrange(5,9) for _ in range(d)]
            if sum(l)>8 or sum(h)<8:continue
            vertices=simplex_vertices(l,h,8)
            scores=[random.randrange(-5,10) for _ in range(d)]
            actual=[sum(Fraction(p*s,8) for p,s in zip(x,scores)) for x in vertices]
            pl=[I(p)/8 for p in l];pu=[I(p)/8 for p in h]
            for maximum,expect in ((True,max(actual)),(False,min(actual))):
                enclosure=box_minmax([I(x) for x in scores],pl,pu,maximum)
                assert rational_contains(enclosure,expect),(d,l,h,scores,enclosure,expect)
            successes+=1
    return successes

def check_exp_posteriors():
    rep=Replay(55,False)
    box=json.loads((Path(__file__).parent/"data/outer_boxes_independent.json").read_text())["boxes"][-1]
    result=rep.bayes_kernel_tests(box)
    assert result['bayes_checks']>=70
    for a in (1,2):
        for s in (0,1):
            for m in range(4):
                w=rep.w[s][a][m]
                assert certain_lt(Z,w) and certain_lt(w,U)
    return result

if __name__=='__main__':
    out={'lp_test_cases':check_lp(),'bayesian':check_exp_posteriors(),
         'scope':'Consistency checks only; full invariant-box and weighted gain are established in independent replay runs.'}
    (Path(__file__).parent/'independent_unit_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
