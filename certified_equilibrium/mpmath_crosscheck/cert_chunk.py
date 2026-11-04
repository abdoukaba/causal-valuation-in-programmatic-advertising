import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
import sys,time,json
from run_hjb_certificate import C2,upper,D
start,end=int(sys.argv[2]),int(sys.argv[3]);path=sys.argv[1]
c=C2(path); records=[];time0=time.monotonic()
for k in range(start,end):
 tops=[]
 for j in range(10):
  pc=D(2*j+1)/D(20)
  ub=upper(c,k,pc,D('.05'),D('.5'),D('.5'))
  tops.append(ub);records.append({'k':k,'j':j,'upper':ub})
 print(k,max(tops),round(time.monotonic()-time0,2),flush=True)
eta='eta60' if '0.60' in path else 'eta50'
out=_os.path.join(HERE,'outputs',f'hjb_{eta}_{start}_{end}.json')
open(out,'w').write(json.dumps(records))
print('SAVED',out)
