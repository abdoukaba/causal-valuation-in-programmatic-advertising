"""Check the outcomes used in the paper including the negative tests."""
import json
import sys
from decimal import Decimal as D
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/'results'/name).read_text())
mode=sys.argv[1]
if mode in ('grid','all'):
    wide=read('cert_21_60_r9_p224.json')
    assert wide['invariant_and_contracting'] and D(wide['jacobian_norm_upper'])<D('.996801')
    assert wide['multicell_events']==103 and read('mpmath_center.json')['passed']
    assert D(read('cert_21_60_r28_p128.json')['root_tv_distance_upper'])<D('1.262e-14')
if mode=='all':
    assert not read('cert_21_60_r8_p128.json')['invariant_and_contracting']
    assert read('cert_31_120_r10_p128.json')['invariant_and_contracting']
    assert read('replay_ipinyou_30.json')['numerator']==4676772289 and read('replay_ipinyou_45.json')['numerator']==7900322858
if mode in ('continuous','all'):
    assert D(read('continuous_dual_160_160_p160.json')['exact_response_residual_TV_upper'])<D('.062327')
    assert not any(read(f'continuous_tube_{tol}_40.json')['sufficient_gain_below_one'] for tol in (1000,10000))
    tr=read('transfer.json')
    assert all(D(tr['linear_'+m]['extended_flow_continuous_residual_TV_upper'])<D('.062785') for m in ('FEM','PINN'))
print('All requested checks completed.')
