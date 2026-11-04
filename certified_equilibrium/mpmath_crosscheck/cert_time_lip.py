import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
import sys,json,math
from iv_policy import C2,D2,O,box,pol_jets,norm_ub,D
c=C2(sys.argv[1]);rows=[];globalmax=0
for k in range(60):
 jets=pol_jets(c,k,D2('.5'),D2(box(D(0),D(1)),u=O))
 starts=(0,2,5,8);counts=(2,3,3,3)
 bounds=[math.nextafter(60*sum(math.nextafter(norm_ub(x.u),math.inf) for x in jets[start:start+count]),math.inf) for start,count in zip(starts,counts)]
 mx=max(bounds);globalmax=max(globalmax,mx);rows.append({'k':k,'bound':mx,'by_budget':bounds})
print('MAX TIME LIP',globalmax,'cell',max(rows,key=lambda x:x['bound']))
open(_os.path.join(HERE,'outputs','time_lip_'+('eta60' if '0.60' in sys.argv[1] else 'eta50')+'.json'),'w').write(json.dumps({'max':globalmax,'per_cell':rows},indent=2))
