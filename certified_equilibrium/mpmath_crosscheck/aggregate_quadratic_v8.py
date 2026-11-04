import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
import glob,json
from decimal import Decimal as D,Context,localcontext,ROUND_CEILING
root=_os.path.join(HERE,'outputs')+_os.sep
with localcontext(Context(prec=55,rounding=ROUND_CEILING)):
 out={}
 for label in ('50','60'):
  tag='eta'+label
  H=[];L=[]
  for p in glob.glob(root+f'hjb_{tag}_*.json'):H+=json.load(open(p))
  for p in glob.glob(root+f'lip_{tag}_*.json'):L+=json.load(open(p))
  assert len(H)==len(L)==600
  for rows in (H,L):assert set((x['k'],x['j']) for x in rows)==set((i,j) for i in range(60) for j in range(10))
  h_orig=max(D(x['upper']) for x in H)
  l_orig=max(D(x['upper']) for x in L)
  tl_orig=D(str(json.load(open(root+f'time_lip_{tag}.json'))['max']))
  ref_orig=D(str(json.load(open(root+f'forward_{tag}_50_60.json'))['max_res']))
  cert=json.load(open(root+f'quadratic_eta{label}_4000.json'))
  cw_orig=D(cert['weighted_upper_bound']);c=D('0.805397') if label=='50' else D('0.969303')
  assert c>cw_orig
  dh=D('0.0000121') if label=='50' else D('0.0000122')
  Lp=D('.173') if label=='50' else D('.174')
  Lt=D('.6');A=D('.0029')
  ref_grid=D('.002141') if label=='50' else D('.002147')
  assert dh>h_orig and Lp>l_orig and Lt>tl_orig and ref_grid>ref_orig
  lam=D('2.25');eps=D('.05');N=D(6000);hh=D(1)/N
  poisson_tail=N*(lam*hh)**3/D(6)
  freeze=lam*Lt*hh/D(4)
  within=(2*lam+Lt)*hh
  ref=ref_grid+poisson_tail+freeze+within
  belief=D('1.1')*Lp*A.sqrt()
  pol=D('1.42')*dh/eps
  total=ref+belief+pol
  dist=total/(D(1)-c)
  out[tag]={'source_uniform_upper_bounds':{'HJB_residual':str(h_orig),'belief_Lipschitz':str(l_orig),'time_Lipschitz':str(tl_orig),'reference_node_residual':str(ref_orig),'weighted_contraction':str(cw_orig)},
   'safe_bounds':{'HJB':str(dh),'L_p':str(Lp),'L_time':str(Lt),'quadratic_variation':str(A),'c_weighted':str(c),'node_error':str(ref_grid)},
   'terms':{'Poisson_tail':str(poisson_tail),'midpoint_policy_freezing':str(freeze),'within_step_drift':str(within),'ref_error':str(ref),'belief_coupling':str(belief),'policy_error':str(pol),'response_residual':str(total)},
   'final_equilibrium_error':str(dist),'safe_display':'< 0.071' if label=='50' else '< 0.45'}
 print(json.dumps(out,indent=2))
 open(root+'full_quadratic_certificates_v8.json','w').write(json.dumps(out,indent=2))
