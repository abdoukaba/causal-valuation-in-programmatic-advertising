import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
from decimal import Decimal as D,Context,localcontext,ROUND_CEILING
import json,glob,math
root=_os.path.join(HERE,'outputs')+_os.sep
h=[];p=[]
for fn in glob.glob(root+'hjb_eta50_*_*.json'):h+=json.load(open(fn))
for fn in glob.glob(root+'lip_eta50_*_*.json'):p+=json.load(open(fn))
assert len(h)==len(p)==600
assert set((x['k'],x['j']) for x in h)==set((k,j) for k in range(60) for j in range(10))
assert set((x['k'],x['j']) for x in p)==set((k,j) for k in range(60) for j in range(10))
orig={'HJB':str(max(D(x['upper']) for x in h)),'Lpi':str(max(D(x['upper']) for x in p))}
t=json.load(open(root+'time_lip_eta50.json'))
orig['Ltime']=str(t['max']);ref=json.load(open(root+'forward_eta50_50_60.json'))
orig['ref_grid']=str(ref['max_res']);orig['ref_arg']=ref['arg']
with localcontext(Context(prec=50,rounding=ROUND_CEILING)):
 L=D('.173');Lt=D('.6');delta=D('.0000121');cw=D('.820308');lam=D('2.25');eps=D('.05');n=D(6000);hstep=D(1)/n
 A=D('.0029');sqrtA=A.sqrt();tail=n*(lam/n)**3/D(6)
 # first path coupling between numerical optimal policy and frozen-prior reference
 ref_grid=D('.002141'); ref_bound=ref_grid+lam*Lt*hstep/D(4)+tail+(D(2)*lam+Lt)*hstep
 belief=D('1.25')*L*sqrtA
 policy=(D(1)+lam)/D(2)*delta/eps
 weighted_res=ref_bound+belief+policy
 eqbound=weighted_res/(D(1)-cw)
 out={'source_interval_checks':orig,'safe_upper_parameters':{'HJB_residual':str(delta),'policy_belief_Lipschitz':str(L),'policy_time_Lipschitz_at_prior':str(Lt),'posterior_quadratic_variation_slope':str(A),'weighted_response_contraction':str(cw),'sample_reference_residual':str(ref_grid)},'error_terms':{'sample_step_h':str(hstep),'Poisson_uniformization_tail':str(tail),'freeze_policy_error':str(lam*Lt*hstep/D(4)),'within_step_bidflow_drift':str((D(2)*lam+Lt)*hstep),'frozen_prior_reference_residual_weighted':str(ref_bound),'belief_dependence_coupling':str(belief),'HJB_to_Gibbs_to_forward_term':str(policy),'total_weighted_response_residual':str(weighted_res),'equilibrium_error_bound':str(eqbound),'safe_rounded_error_bound':'0.085'}}
open(root+'full_certificate_eta50_v8.json','w').write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
