import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads((ROOT/path).read_text())
def draw():
    w=read('data/fem_path.json');q=np.array(w['nodal_flow'])/w['scale']
    c=read('results/center_21_60.json');x=np.array(c['center'])/c['scale']
    x=np.column_stack([x,1-x.sum(axis=1)])
    fig,(a,b)=plt.subplots(1,2,figsize=(9.4,3.4),layout='constrained')
    colors=['#295772','#ad512f','#337b63']
    for j,label in enumerate(['Abstain','Low bid','High bid']):
        a.plot(np.linspace(0,1,len(q)),q[:,j],c=colors[j],label=label,lw=2.1)
        i=np.arange(0,60,5)
        a.plot(i/60,x[i,j],'o',color=colors[j],mfc='white',mew=1.0,ms=3.8)
    a.set(xlabel='Time',ylabel='Population bid share',xlim=(-.02,1.02),ylim=(.16,.50),
          title='Bid paths on the unit horizon')
    a.legend(frameon=False,ncol=1,loc='center left',bbox_to_anchor=(.02,.40),fontsize=8)
    labels=['Prior','20 / 24','40 / 48','80 / 96','160 / 160']
    vals=[read('data/certificate_fem_composite_0.45.json')['step_residual_TV']]
    for g,p in [(20,24),(40,48),(80,96),(160,160)]:
        vals.append(float(read(f'results/continuous_dual_{g}_{p}_p160.json')['exact_response_residual_TV_upper']))
    b.bar(range(5),vals,color=['#888e92']+['#38765c']*4,width=.65)
    for k,v in enumerate(vals):b.text(k,v+.008,f'{v:.3f}',ha='center',fontsize=8)
    b.set(xticks=range(5),xticklabels=labels,ylim=(0,.43),xlabel='Output blocks / belief intervals',
          ylabel='Upper bound in total variation',title='Continuous best-response residual')
    for ax in (a,b):
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',lw=.5,color='#d8e0e2');ax.set_axisbelow(True)
    fig.savefig(ROOT/'comparison.pdf',bbox_inches='tight');plt.close(fig)
if __name__=='__main__':draw()
