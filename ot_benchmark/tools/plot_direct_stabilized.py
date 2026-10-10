"""Grouped complete-time comparisons; each group contains six real pairs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'validation/2026-10-06'
fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
labels=['movielens_reg_0.01','movielens_reg_0.001','kos_reg_0.01','kos_reg_0.001']
for ax,name,title in zip(axes,['direct_stabilized_dev_round21','direct_stabilized_validation_round21'],['Previously viewed pairs','New disjoint pairs']):
    j=json.loads((ROOT/(name+'_audit.json')).read_text())
    for i,(method,label) in enumerate([('direct_stabilized','Direct, block 5000'),('direct_stabilized_250','Direct, block 250')]):
        y=[j['groups'][k]['totals'][method]['sum_medians']/j['groups'][k]['totals']['guarded']['sum_medians'] for k in labels]
        ax.bar(np.arange(4)+(i-.5)*.32,y,.32,label=label)
    ax.axhline(1,color='black',linestyle='--',linewidth=1)
    ax.set_xticks(np.arange(4),['MovieLens\n.01','MovieLens\n.001','KOS\n.01','KOS\n.001'])
    ax.set_title(title);ax.set_xlabel('Corpus / regularization');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
axes[0].set_ylabel('Direct time / guarded time\n(ratio of sums of case medians)')
fig.suptitle('Guarded switching versus direct stabilized baselines')
fig.text(.5,.01,'3 repeats per condition; shared preprocessing excluded; descriptive timing, not statistical significance.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,.93))
for ext in ['png','pdf']:fig.savefig(ROOT/f'direct_stabilized_round21_comparison.{ext}',dpi=180)
