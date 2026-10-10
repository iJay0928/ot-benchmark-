"""Reporting tests use fabricated rows only, never selected formal seed outcomes."""
import itertools
import json
from pathlib import Path
import pytest
from core import METHODS, write_json
import summarize_results

def fake_rows():
    rows=[]
    for seed,n,d,e,m,repeat in itertools.product((210101,210102,210103),(16,64),
            (0.,1e-5,1e-4,1e-3),(.01,.001),METHODS,(1,2,3)):
        status='SUCCESS' if d==0 else 'QUALITY_FAILED'
        t=(1,2,4)[METHODS.index(m)]*repeat if status=='SUCCESS' else 29.
        rows.append(dict(task_id=f'fake_{len(rows)}',seed=seed,n=n,delta=d,epsilon=e,
                         method=m,repeat=repeat,status=status,elapsed_seconds=t,native_stop_met=False))
    return rows

def setup_data(out,rows):
    write_json(out/'results/summary.json',rows)
    write_json(out/'protocol.json',dict(n=[16,64],delta=[0.,1e-5,1e-4,1e-3],
        epsilon=[.01,.001],seeds=[210101,210102,210103]))

def test_complete_denominators_and_paired_only(tmp_path,monkeypatch):
    setup_data(tmp_path,fake_rows());monkeypatch.setattr(summarize_results,'ROOT',tmp_path)
    summarize_results.summarize(tmp_path)
    stats=json.loads((tmp_path/'results/overview.json').read_text(encoding='utf-8'))
    assert stats['calls']==432 and stats['method_conditions']==144
    assert stats['all_three_success_conditions']==12
    assert stats['comparisons'][0]['median_left_over_right']==.5
    assert stats['comparisons'][0]['paired_conditions']==12
    assert stats['condition_counts'][METHODS[0]]=={'SUCCESS':12,'QUALITY_FAILED':36}

def test_unstable_excluded_from_speed(tmp_path,monkeypatch):
    rows=fake_rows();rows[0]['status']='TIMEOUT';rows[0]['elapsed_seconds']=None
    setup_data(tmp_path,rows);monkeypatch.setattr(summarize_results,'ROOT',tmp_path)
    summarize_results.summarize(tmp_path)
    stats=json.loads((tmp_path/'results/overview.json').read_text(encoding='utf-8'))
    assert stats['all_three_success_conditions']==11
    assert stats['condition_counts'][METHODS[0]]['UNSTABLE']==1
    conditions=json.loads((tmp_path/'results/conditions.json').read_text(encoding='utf-8'))
    unstable=[c for c in conditions if c['status']=='UNSTABLE']
    assert len(unstable)==1 and unstable[0]['median_seconds'] is None

def test_partial_report_rejected(tmp_path):
    setup_data(tmp_path,fake_rows()[:-1])
    with pytest.raises(RuntimeError,match='432'):summarize_results.summarize(tmp_path)
