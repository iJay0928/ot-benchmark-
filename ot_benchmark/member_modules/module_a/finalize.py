"""Post-run only: tests, independent audit, statistics, EDA and figure QA."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from core import ROOT, write_json
from audit_results import audit
from summarize_results import summarize
from plot_results import plot

def main(out,style_tools):
    result=subprocess.run([sys.executable,'-m','pytest',str(ROOT/'test_module_a.py'),
                           str(ROOT/'test_reporting.py'),'-q'],capture_output=True,text=True)
    (out/'tests.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    print(result.stdout)
    if result.returncode:raise RuntimeError('module tests failed')
    audit(out);summarize(out)
    if style_tools:
        sys.path.insert(0,str(style_tools))
        from profile_data import profile_data,render_report
        profile=profile_data(str(out/'results/summary.csv'),group_cols=['method','n','delta','epsilon'])
        (out/'results/data_profile.md').write_text(render_report(profile),encoding='utf-8')
        write_json(out/'results/data_profile.json',profile)
    plot(out,style_tools)
    if style_tools:
        from check_figure import check_figure
        checks=[]
        for p in (out/'results/figures').iterdir():
            if p.suffix not in ('.png','.svg','.pdf') or p.stem.endswith('_preview'):continue
            issues,info=check_figure(str(p),min_dpi=300)
            checks.append(dict(file=p.name,issues=issues,info=info))
        write_json(out/'results/figures/file_audit.json',checks)
        if any(level=='FAIL' for c in checks for level,msg in c['issues']):
            raise RuntimeError('figure file audit failed')
    print('Ready for visual inspection; package only after that review.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--style-tools',type=Path);a=p.parse_args();main(a.output,a.style_tools)
