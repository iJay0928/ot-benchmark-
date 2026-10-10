"""Plot descriptive grouped complete-time ratios from the frozen scale audit."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'validation/2026-10-06'
fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
for ax,name,title in zip(axes,['quality_scale_round15','quality_scale_validation_round15'],['Exploratory seeds','New validation seeds']):
    j=json.loads((ROOT/(name+'_audit.json')).read_text());groups=j['groups'];x=np.arange(len(groups))
    methods=[m for m in j['totals'] if m!='residual_gate'];width=.24
    for index,m in enumerate(methods):
        y=[g['residual_gate']['sum_medians']/g[m]['sum_medians'] for g in groups.values()]
        ax.bar(x+(index-(len(methods)-1)/2)*width,y,width,label=m.replace('every_','Every '),
               color={'every_50':'#1f77b4','every_250':'#ff7f0e','every_1000':'#2ca02c'}[m])
    ax.axhline(1,color='black',linewidth=1,linestyle='--')
    ax.set_xticks(x,[f'{n}\n{reg}' for n in [30,100,200] for reg in ['.01','.001']])
    ax.set_xlabel('Active support size per side / regularization')
    ax.set_title(title);ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Baseline time / periodic time\n(ratio of sums of case medians)')
fig.suptitle('Quality-controlled stopping: inspection frequency and scale')
fig.text(.5,.01,'3 seeds per group, 3 timing repeats; descriptive results, dependent groups; common quality gate.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.94))
for ext in ['png','pdf']:fig.savefig(ROOT/f'quality_scale_round15_comparison.{ext}',dpi=180)
