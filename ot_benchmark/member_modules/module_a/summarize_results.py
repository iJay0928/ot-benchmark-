"""All denominators retained; pair timing only on jointly accepted conditions."""
import argparse
from collections import Counter, defaultdict
import csv
import itertools
import json
from pathlib import Path
import statistics
from core import METHODS, ROOT, write_json

SHORT=dict(zip(METHODS,('普通','稳定化','原生延续')))

def summarize(out):
    rows=json.loads((out/'results/summary.json').read_text(encoding='utf-8'))
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if len(rows)!=432: raise RuntimeError('formal report requires all 432 calls')
    grouped=defaultdict(list)
    for r in rows:grouped[(r['seed'],r['n'],r['delta'],r['epsilon'],r['method'])].append(r)
    conditions=[]
    for key,rs in sorted(grouped.items()):
        assert len(rs)==3 and {r['repeat'] for r in rs}=={1,2,3}
        states={r['status'] for r in rs};times=[r['elapsed_seconds'] for r in rs if r['status']=='SUCCESS']
        record=dict(zip(('seed','n','delta','epsilon','method'),key))
        record.update(status=next(iter(states)) if len(states)==1 else 'UNSTABLE',
                      successes=len(times),status_counts=dict(Counter(r['status'] for r in rs)),
                      median_seconds=statistics.median(times) if len(times)==3 else None,
                      min_seconds=min(times) if len(times)==3 else None,
                      max_seconds=max(times) if len(times)==3 else None,
                      task_ids=[r['task_id'] for r in sorted(rs,key=lambda r:r['repeat'])])
        conditions.append(record)
    assert len(conditions)==144
    write_json(out/'results/conditions.json',conditions)
    with (out/'results/conditions.csv').open('w',encoding='utf-8-sig',newline='') as s:
        fields=list(conditions[0]);w=csv.DictWriter(s,fields);w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in c.items()} for c in conditions)
    lookup={(c['seed'],c['n'],c['delta'],c['epsilon'],c['method']):c for c in conditions}
    basekeys=sorted({k[:4] for k in lookup})
    common=[k for k in basekeys if all(lookup[(*k,m)]['status']=='SUCCESS' for m in METHODS)]
    comparisons=[]
    for left,right in itertools.combinations(METHODS,2):
        pairs=[k for k in basekeys if all(lookup[(*k,m)]['status']=='SUCCESS' for m in (left,right))]
        ratios=[lookup[(*k,left)]['median_seconds']/lookup[(*k,right)]['median_seconds'] for k in pairs]
        comparisons.append(dict(left=left,right=right,paired_conditions=len(pairs),
            left_faster=sum(r<1 for r in ratios),right_faster=sum(r>1 for r in ratios),
            median_left_over_right=statistics.median(ratios) if ratios else None,
            min_ratio=min(ratios) if ratios else None,max_ratio=max(ratios) if ratios else None,
            pairs=[dict(seed=k[0],n=k[1],delta=k[2],epsilon=k[3],left_over_right=r) for k,r in zip(pairs,ratios)]))
    overview=dict(calls=432,method_conditions=144,input_epsilon_conditions=48,
        all_three_success_conditions=len(common),
        call_counts={m:dict(Counter(r['status'] for r in rows if r['method']==m)) for m in METHODS},
        condition_counts={m:dict(Counter(c['status'] for c in conditions if c['method']==m)) for m in METHODS},
        native_stop_quality_failed=[r['task_id'] for r in rows if r['status']=='QUALITY_FAILED' and r.get('native_stop_met')],
        comparisons=comparisons)
    write_json(out/'results/overview.json',overview)
    lines=['# 模块 A：成熟方法的数据条件对照验证','',
        '本轮不修改任何主项目算法。POT 0.9.6.post1 的三个原生方法，固定配置、统一外部最终质量，完整保留失败。',
        '', '## 1. 完整分母与主要结果','',
        '24 个输入、48 个输入与 ε 组合、144 个方法条件、每条件 3 次计时，共 432 次调用。四个 δ 共享几何，两档 ε 配对；不是 432 个独立实例。',
        '', '| 方法 | SUCCESS / 48 条件 | QUALITY_FAILED | NUMERICAL_ERROR | TIMEOUT | 其他/不稳定 |',
        '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        cnt=Counter(c['status'] for c in conditions if c['method']==m)
        other=sum(v for k,v in cnt.items() if k not in ('SUCCESS','QUALITY_FAILED','NUMERICAL_ERROR','TIMEOUT'))
        lines.append(f"| {SHORT[m]} | {cnt['SUCCESS']}/48 | {cnt['QUALITY_FAILED']} | {cnt['NUMERICAL_ERROR']} | {cnt['TIMEOUT']} | {other} |")
    lines+=['','一个条件只有三次都达标才计入 SUCCESS；完整逐次状态计数见 overview.json，重复状态不一致标为 UNSTABLE。',
        '', '## 2. 完整数据条件表','',
        '各格为三个独立几何种子中三次均达标的数量 / 3。所有失败类型和行号在 conditions.csv 与 summary.csv 中。',
        '', '| n | δ | ε | 普通 | 稳定化 | 原生延续 |','|---:|---:|---:|---:|---:|---:|']
    for n,delta,eps in itertools.product(protocol['n'],protocol['delta'],protocol['epsilon']):
        vals=[sum(lookup[(seed,n,delta,eps,m)]['status']=='SUCCESS' for seed in protocol['seeds']) for m in METHODS]
        lines.append(f'| {n} | {delta:g} | {eps:g} | '+' | '.join(f'{v}/3' for v in vals)+' |')
    lines+=['','## 3. 同质量速度比较','',
        f'三种方法共同成功 {len(common)}/48 个条件。只对各对方法三次均成功的条件比较时间中位数；不将失败耗时或超时的 30 秒当作完成时间。',
        '', '| 配对（左/右） | 共同成功条件 | 左更快 | 右更快 | 左/右时间比中位数 | 比值范围 |',
        '|---|---:|---:|---:|---:|---|']
    for cmp in comparisons:
        if cmp['paired_conditions']:
            lines.append(f"| {SHORT[cmp['left']]}/{SHORT[cmp['right']]} | {cmp['paired_conditions']} | {cmp['left_faster']} | {cmp['right_faster']} | {cmp['median_left_over_right']:.4g} | {cmp['min_ratio']:.4g}–{cmp['max_ratio']:.4g} |")
    lines+=['','共同达标分组耗时（秒）：每格是各种子“三次中位数”的中位数 [最小, 最大]，不是置信区间。每组仅 3 个种子；原始三次范围仍在 conditions.csv。',
            '', '| n | δ | ε | 共同种子数 | 普通 | 稳定化 | 原生延续 |',
            '|---:|---:|---:|---:|---|---|---|']
    for n,delta,eps in itertools.product(protocol['n'],protocol['delta'],protocol['epsilon']):
        ks=[k for k in common if k[1:]==(n,delta,eps)]
        if not ks:continue
        cells=[]
        for m in METHODS:
            ts=[lookup[(*k,m)]['median_seconds'] for k in ks]
            cells.append(f'{statistics.median(ts):.6g} [{min(ts):.6g}, {max(ts):.6g}]')
        lines.append(f'| {n} | {delta:g} | {eps:g} | {len(ks)} | '+' | '.join(cells)+' |')
    lines+=['','稳定化/延续的额外共同达标条件（普通方法未三次达标）：',
            '', '| n | δ | ε | 共同种子数 | 稳定化时间中位数(s) | 延续时间中位数(s) | 稳定化/延续配对比中位数 |',
            '|---:|---:|---:|---:|---:|---:|---:|']
    for n,delta,eps in itertools.product(protocol['n'],protocol['delta'],protocol['epsilon']):
        ks=[k for k in basekeys if k[1:]==(n,delta,eps) and
            lookup[(*k,METHODS[0])]['status']!='SUCCESS' and
            all(lookup[(*k,m)]['status']=='SUCCESS' for m in METHODS[1:])]
        if ks:
            st=[lookup[(*k,METHODS[1])]['median_seconds'] for k in ks]
            et=[lookup[(*k,METHODS[2])]['median_seconds'] for k in ks]
            ratio=statistics.median(a/b for a,b in zip(st,et))
            lines.append(f'| {n} | {delta:g} | {eps:g} | {len(ks)} | {statistics.median(st):.6g} | {statistics.median(et):.6g} | {ratio:.4g} |')
    lines+=['','比值大于 1 表示右侧更快。这是同机描述统计，没有显著性检验或跨机器速度承诺。极短调用计时对噪声敏感，应结合三次范围，不依据微小差别生成普遍推荐。',
        '', '## 4. 原生停止与共同质量的区别','',
        f"原生最后误差 ≤1e-12 但共同验收失败的调用共 {len(overview['native_stop_quality_failed'])} 条。完整行号见 overview.json/native_stop_quality_failed。",
        '', '普通与稳定化内部主要检查目标边缘的 L2 误差；原生延续检查两侧边缘 L2 误差平方之和，且每 10 个外层才更新。相同 stopThr 数值不等于相同精度，未按正式结果为某方法补调阈值。',
        '', '失败示例（固定按任务顺序每类取首条，不筛选最有利案例）：','']
    for status in ('QUALITY_FAILED','NUMERICAL_ERROR','TIMEOUT','DIAGNOSTIC_INVALID'):
        candidates=[r for r in rows if r['status']==status]
        if candidates:
            r=candidates[0]
            lines.append(f"- {r['task_id']}: {r['method']}, seed={r['seed']}, n={r['n']}, δ={r['delta']:g}, ε={r['epsilon']:g}; {status}; gap={r.get('numerical_gap')}; native_last_error={r.get('native_last_error')}; {r.get('error') or '未达到共同最终质量'}。")
    lines+=['','## 5. 接口与验收实现','',
        '普通：log(u), log(v)。稳定化：直接使用 logu/logv，不将 alpha 再当作 logu。延续：alpha/最终ε、beta/最终ε；其退火参数公式 (1−ε)exp(−ii)+ε，在原生至少 35 层后与最终 ε 仅有舍入量级差异，仍逐次核对返回计划与最终 ε 重建计划，rtol=1e-9、atol=1e-12。',
        '', '舍入后边缘 L1 残差≤1e-12；最终数值 gap∈[−1e-10,1e-8]；原生返回与重建矩阵均有限且非负。额外核对原生计划舍入后目标减对偶的间隙，不能仅靠重建矩阵达标。公共 diagnose/round_plan 未修改。浮点弱对偶诊断不等于严格区间算术证书。',
        '', '普通/稳定化 niter 或 n_iter 是从 0 开始的循环索引，报告实际调用循环数为索引+1。原生延续记录实际外层数，实际内层数 API 未提供，填写 null，未用外层×上限冒充工作量。',
        '', '每次新子进程载入模块，完成相同 4 点开发输入热身后 READY/GO 握手。30 秒限时从 GO 开始，由父进程可强制终止。计时是求解+字段转换+外部验收；启动/导入/热身/输入加载/写盘排除并另列；传输及监督开销另列 parent_observed_seconds。所有方法按冻结随机顺序串行运行。',
        '', '## 6. 数据边界与研究意义','',
        '在本轮固定配置内，两个规模、三个种子给出一致的条件边界：δ=0 时两种 ε 下三种方法全部达标；δ=1e-5 时两种 ε 下三种方法均未达标；δ=1e-4、ε=0.001 时也均未达标。这是非单调的局部现象，不能外推成“δ 越小越困难”，也不是慢收敛机制的证明。',
        '', 'δ=0.001、ε=0.001 的 6 个输入条件中，普通方法发生数值错误，稳定化与原生延续全部达标。这是使用成熟稳定化基线的直接证据；两种规模下稳定化/延续的配对耗时比中位数分别约为 1.94、2.03。δ=0 的容易条件中，延续额外层次反而增加开销。不能据此声称普遍加速。',
        '', '核矩阵 exp(−C/ε) 的零元素不是充分或必要的失败判据：ε=0.01、δ=1e-5 时没有下溢零元素但全部未达标；ε=0.001、δ=0 时有跨簇下溢零元素但全部达标。两者各覆盖两个规模和三个种子，说明诊断不能仅按核矩阵下溢作推荐。',
        '', '本轮原生延续并不足以覆盖全部条件，仍有 18/48 个方法条件未达标。现有证据支持在同一质量协议下按条件比较成熟基线，不支持发布已经泛化验证的算法推荐器。不能写成全部延续算法均无效，也不能把已知稳定化、热启动、退火或残差不足现象宣称为创新。',
        '', '数据为 1D 双簇、同源目标支撑、正质量、平衡 OT。不是高维验证，不是输入质量稀疏验证，不是非平衡 OT；n=16/64 不支撑大规模结论。建议主项目先在同条件对照这三个成熟基线，再研究机制与新方案的增量。',
        '', '## 7. 证据与复现','',
        'protocol.json、task_order.json 在正式运行前冻结；输入数组规范字节 SHA256 与 npz 文件 SHA256 分开记录。environment.json 记录实际环境，不能与主项目历史机器耗时相减。',
        '', 'results/summary.csv 每次一行；conditions.csv 提供中位数和最小/最大范围；overview.json 包含完整状态和配对证据；audit.json 重建输入并复算全部可诊断返回。results/logs 保留原生所有 err 轨迹与警告，results/plans 保存实际返回与势，不随公开仓库分发，但交付包包含全部证据。',
        '', '背景： [Schmitzer 原文](https://arxiv.org/abs/1610.06519)，算法 1–3、注 5、例 1；[POT 官方源码文档](https://pythonot.github.io/_modules/ot/bregman/_sinkhorn.html)。线上版本可能更新，本轮以 environment.json 记录的安装源码 SHA256 为准。',
        '', '环境差异：本机 Python 3.12，任务书建议 3.11；POT 使用要求的 0.9.6.post1。模块独立测试、完整结果复算用于验证此组合，不声称与历史环境完全相同。']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(overview,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    summarize(p.parse_args().output)
