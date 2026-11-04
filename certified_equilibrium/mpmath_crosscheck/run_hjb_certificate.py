"""Uniform interval Taylor enclosure of HJB residual for an EXACT DECIMAL
piecewise cubic-time/Chebyshev-belief candidate. Defines candidate solely from
candidate_etaXX_n7.json; no reliance on the accuracy of the spectral integrator.

All transcendental functions and automatic differentiation are in mpmath.iv.
Final decimal outputs are inflated by 1e-12 to allow for any decimal formatting.
"""
import json,sys,time,math
from decimal import Decimal as D, Context, localcontext, ROUND_CEILING
from iv_taylor import C2,D2,O,box

def mag_up(x):
 return math.nextafter(max(abs(float(x.a)),abs(float(x.b))), math.inf)

def upper(c,k,pc,rp,uc,ru):
 mid=c.residualJ(k,D2(str(pc),p=O),D2(str(uc),u=O))
 big=c.residualJ(k,D2(box(pc-rp,pc+rp),p=O),D2(box(uc-ru,uc+ru),u=O))
 with localcontext(Context(prec=50,rounding=ROUND_CEILING)):
  P=D(str(rp));U=D(str(ru))
  out=[]
  for r,b in zip(mid,big):
   val=(D(str(mag_up(r.v)))+P*D(str(mag_up(r.p)))+U*D(str(mag_up(r.u)))+
        P*P*D(str(mag_up(b.pp)))/2+P*U*D(str(mag_up(b.pu)))+
        U*U*D(str(mag_up(b.uu)))/2)
   out.append(str(+val))
 return max(out,key=lambda z:D(z))

def run(path, output, nbelief=10, ntime=1):
 c=C2(path); rows=[]; top=D(0); start=time.monotonic()
 for k in range(60):
  celltop=D(0)
  for j in range(nbelief):
   # Binary rational when nbelief=10? Use exact Decimal for partition; represent decimal strings
   for i in range(ntime):
    with localcontext(Context(prec=50)):
     pc=(D(2*j+1)/D(2*nbelief));rp=(D(1)/D(2*nbelief))
     uc=(D(2*i+1)/D(2*ntime));ru=(D(1)/D(2*ntime))
    ub=upper(c,k,pc,rp,uc,ru)
    b=D(ub)
    if b>top:top=b
    if b>celltop:celltop=b
    rows.append({'k':k,'p_cell':j,'u_cell':i,'upper':ub})
  print('time_cell',k,'max',str(celltop),'overall',str(top),'elapsed',round(time.monotonic()-start,1),flush=True)
 res={'source_candidate':path,'method':'mpmath.iv arbitrary-precision interval automatic differentiation up to order two, centered Taylor remainder on each rectangle, upper endpoints then rounded upwards using nextafter and Decimal ROUND_CEILING','belief_subintervals':nbelief,'time_subintervals_per_cell':ntime,'time_cells':60,'reported_sup_upper':str(top),'safety_enlarged_upper':str(top+D('1e-12')),'boxes':len(rows),'per_box':rows}
 open(output,'w').write(json.dumps(res,indent=2))
 print('CERTIFIED',res['safety_enlarged_upper'],flush=True)
if __name__=='__main__':
 run(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 10,int(sys.argv[4]) if len(sys.argv)>4 else 1)
