import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
import sys,json,time
from decimal import Decimal as D
from iv_policy import C2,lip_bound
src=sys.argv[1];start=int(sys.argv[2]);end=int(sys.argv[3]);c=C2(src)
rows=[];start_t=time.time()
for k in range(start,end):
 vals=[]
 for j in range(10):
  ub,byb=lip_bound(c,k,D(2*j+1)/D(20),D('.5'),D('.05'),D('.5'))
  vals.append(ub);rows.append({'k':k,'j':j,'upper':str(ub),'by_budget':[str(x) for x in byb]})
 print(k,max(vals),round(time.time()-start_t,2),flush=True)
eta='eta60' if '0.60' in src else 'eta50'
out=_os.path.join(HERE,'outputs',f'lip_{eta}_{start}_{end}.json');open(out,'w').write(json.dumps(rows));print('Saved',out)
