import json, time
from pathlib import Path
from independent_interval_replay import Replay
start=time.monotonic()
r=Replay(55,False)
box=([0,0,0],[1000000,1000000,1000000]);boxes=[box]
out=Path(__file__).parent/'data/outer_boxes_independent.json'
for k in range(8):
    st=r.baseline(box,40)
    st,_,_=r.improved(st,box,6)
    low,high=r.restrict(st)
    box=([max(a,b) for a,b in zip(box[0],low)],[min(a,b) for a,b in zip(box[1],high)])
    boxes.append(box)
    out.write_text(json.dumps({'boxes':boxes,'complete':k==7,'elapsed':round(time.monotonic()-start,3)},indent=2)+'\n')
    print('outer',k+1,box,'elapsed',round(time.monotonic()-start,1),flush=True)
print('done',flush=True)
