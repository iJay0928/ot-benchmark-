"""Three non-redundant views: conditional reliability, paired speed, failure types."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from core import METHODS, ROOT, write_json

def plot(out, style_tools=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    if style_tools:
        sys.path.insert(0,str(style_tools))
        from setup_style import setup_style
        from export_figure import export_figure
        from visual_qa import audit_layout,render_preview
        setup_style(journal='general',lang='en',use_sciplots=False)
    else:
        # Portable same-Python fallback when the personal skill tools are absent.
        plt.rcParams.update({'figure.constrained_layout.use':True,'font.size':8,
                             'pdf.fonttype':42,'svg.fonttype':'none'})
        def audit_layout(fig): fig.canvas.draw();return []
        def render_preview(fig,name): fig.savefig(name,dpi=150)
        def export_figure(fig,basename,**kwargs):
            from PIL import Image
            for ext in ('pdf','svg','png'):fig.savefig(str(basename)+'.'+ext,dpi=300)
            im=Image.open(str(basename)+'.png');im.convert('L').save(str(basename)+'_grayscale.png',dpi=(300,300))
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7})
    conditions=json.loads((out/'results/conditions.json').read_text(encoding='utf-8'))
    rows=json.loads((out/'results/summary.json').read_text(encoding='utf-8'))
    lookup={(c['seed'],c['n'],c['delta'],c['epsilon'],c['method']):c for c in conditions}
    keys=sorted({k[:4] for k in lookup});common=[k for k in keys if all(lookup[(*k,m)]['status']=='SUCCESS' for m in METHODS)]
    figs=out/'results/figures';figs.mkdir(exist_ok=True);qa={}
    names=['Knopp','Stabilized','Continuation'];colors=['#0072B2','#D55E00','#999999'];markers=['o','s','^']
    def save(fig,name,size):
        render_preview(fig,str(figs/(name+'_preview.png')))
        issues=audit_layout(fig);qa[name]=issues
        if any(level=='FAIL' for level,msg in issues): raise RuntimeError(issues)
        export_figure(fig,str(figs/name),formats=['pdf','svg','png'],size_inches=size,
                      dpi=300,grayscale_preview=True,tight=False)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.0),constrained_layout=True)
    bins=[(d,e) for d in (0.,1e-5,1e-4,1e-3) for e in (.01,.001)]
    cmap=plt.get_cmap('cividis',4);norm=BoundaryNorm(np.arange(-.5,4.5),4)
    for ax,n in zip(axes,(16,64)):
        vals=np.array([[sum(lookup[(seed,n,d,e,m)]['status']=='SUCCESS' for seed in (210101,210102,210103)) for m in METHODS] for d,e in bins])
        mesh=ax.pcolormesh(np.arange(4),np.arange(9),vals,cmap=cmap,norm=norm,edgecolors='white',linewidth=.5)
        ax.set(xticks=np.arange(3)+.5,xticklabels=names,yticks=np.arange(8)+.5,
               yticklabels=[f'{d:g} / {e:g}' for d,e in bins],ylabel='Cluster mass delta / epsilon',title=f'n = {n}')
        ax.invert_yaxis()
        for i in range(8):
            for j in range(3):
                ax.text(j+.5,i+.5,f'{vals[i,j]}/3',ha='center',va='center',fontsize=8,
                        color='white' if vals[i,j]<2 else 'black')
    fig.colorbar(mesh,ax=axes,ticks=(0,1,2,3),label='Seeds with 3/3 calls accepted',shrink=.8)
    save(fig,'acceptance_matrix',(7.2,4.0))
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.4),constrained_layout=True,sharex=True,sharey=True)
    values=[lookup[(*k,m)]['median_seconds'] for k in common for m in METHODS]
    low=min(values)*.65;high=max(values)*1.5
    for ax,n in zip(axes,(16,64)):
        sub=[k for k in common if k[1]==n]
        for j,m in enumerate(METHODS[1:]):
            for delta in sorted({k[2] for k in sub}):
                pts=[k for k in sub if k[2]==delta]
                ax.scatter([lookup[(*k,METHODS[0])]['median_seconds'] for k in pts],
                           [lookup[(*k,m)]['median_seconds'] for k in pts],
                           color=colors[j+1],marker=('o' if delta==0 else '^' if delta==1e-4 else 's'),
                           s=18,alpha=.8,label=names[j+1] if delta==0 else None)
        ax.plot([low,high],[low,high],color='black',lw=.7,ls='--')
        ax.set(xscale='log',yscale='log',xlim=(low,high),ylim=(low,high),
               xlabel='Knopp median time (s, log)',ylabel='Other method median time (s, log)',
               title=f'n = {n}, common success = {len(sub)}')
        handles=[Line2D([],[],color=colors[j+1],marker='o',linestyle='',label=names[j+1])
                 for j in range(2)]
        handles += [Line2D([],[],color='black',marker=mark,linestyle='',label=f'delta={d:g}')
                    for d,mark in ((0.,'o'),(1e-4,'^'),(1e-3,'s'))
                    if any(k[2]==d for k in sub)]
        ax.legend(handles=handles,frameon=False,fontsize=7,loc='upper left')
    save(fig,'paired_times',(7.2,3.4))
    states=sorted({r['status'] for r in rows})
    counts=np.array([[sum(r['status']==s and r['method']==m for r in rows) for m in METHODS] for s in states])
    fig,ax=plt.subplots(figsize=(7.2,3.0),constrained_layout=True)
    palette=['#0072B2','#E69F00','#999999','#D55E00','#CC79A7','#56B4E9']
    bottoms=np.zeros(3)
    for i,(s,vals) in enumerate(zip(states,counts)):
        ax.barh(np.arange(3),vals,left=bottoms,color=palette[i],label=s,
                edgecolor='white',hatch=('', '//','xx','..','++')[i%5])
        for j,v in enumerate(vals):
            if v:ax.text(bottoms[j]+v/2,j,str(v),va='center',ha='center',fontsize=8)
        bottoms+=vals
    ax.set(yticks=np.arange(3),yticklabels=names,xlim=(0,144),
           xlabel='Repeated calls (144 per method; not independent samples)',title='Outcome composition')
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.25),ncol=3,fontsize=7)
    save(fig,'outcome_types',(7.2,3.0))
    write_json(figs/'layout_audit.json',qa)
    print('Saved three figures with full denominators and paired-only timing')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--style-tools',type=Path);a=p.parse_args();plot(a.output,a.style_tools)
