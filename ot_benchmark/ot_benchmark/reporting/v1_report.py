"""Benchmark v1.0 HTML 报告生成器 (修订版).

支持 within-seed / cross-seed 两级统计、exact/reference/consensus 三种参考类型。
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional


def generate_benchmark_v1_report(
    within_seed: list[dict],
    cross_seed: list[dict],
    env_info: dict,
    output_dir: str,
    *,
    sweep_results: Optional[dict] = None,
    fb_dir: str = "",
    n_scenarios: int = 0,
    n_algorithms: int = 0,
    n_runs: int = 0,
    total_planned: int = 0,
    executed: int = 0,
    success_count: int = 0,
    fail_count: int = 0,
    skip_count: int = 0,
) -> str:
    """生成 v1 HTML 报告。"""
    rate = success_count / max(executed, 1)

    groups: dict[str, list[dict]] = defaultdict(list)
    for s in within_seed:
        groups[s.get("ranking_group", "balanced")].append(s)

    group_labels = {
        "balanced": "Balanced OT 算法",
        "reference": "参考算法 (不参与排名)",
        "unbalanced": "非平衡 OT 算法",
        "baseline": "基线算法",
        "model": "基于模型的算法",
    }

    algo_html = ""
    for g, label in group_labels.items():
        if g not in groups:
            continue
        grp = sorted(groups[g],
                     key=lambda x: x.get("success_rate", 0) or 0,
                     reverse=True)
        algo_html += (
            f'<tr class="gh"><td colspan="8"><b>{label}</b>'
            f' <span class="dim">({g})</span></td></tr>\n')
        for s in grp:
            sr = s.get("success_rate")
            sr_s = f"{sr:.1%}" if sr is not None else "N/A"
            c = ("#22c55e" if (sr or 0) >= 0.9
                 else "#f59e0b" if (sr or 0) >= 0.5
                 else "#ef4444")
            algo_html += (
                f"<tr><td>{s['scenario_id']}</td>"
                f"<td>{s['algorithm']}</td>"
                f"<td>seed={s.get('dataset_seed','')}</td>"
                f"<td style='color:{c}'>{sr_s}</td>"
                f"<td>{_f(s.get('mean_runtime_seconds'))}s</td>"
                f"<td>{_f(s.get('mean_peak_memory_mb'))}MB</td>"
                f"<td>{_f(s.get('mean_ground_truth_error'))}</td>"
                f"<td>{s.get('successful_runs',0)}/{s.get('total_runs',0)}</td></tr>\n")

    # Cross-seed section
    cs_html = ""
    cs_groups: dict[str, list[dict]] = defaultdict(list)
    for cs in cross_seed:
        cs_groups[cs.get("ranking_group", "balanced")].append(cs)
    for g, label in group_labels.items():
        if g not in cs_groups:
            continue
        grp = sorted(cs_groups[g],
                     key=lambda x: x.get("success_rate", 0) or 0,
                     reverse=True)
        cs_html += (
            f'<tr class="gh"><td colspan="8"><b>{label}</b>'
            f' <span class="dim">(跨 seed, {g})</span></td></tr>\n')
        for s in grp:
            sr = s.get("success_rate")
            sr_s = f"{sr:.1%}" if sr is not None else "N/A"
            cs_html += (
                f"<tr><td>{s['scenario_id']}</td>"
                f"<td>{s['algorithm']}</td>"
                f"<td>{s.get('n_seeds','')} seeds</td>"
                f"<td>{sr_s}</td>"
                f"<td>{_f(s.get('mean_runtime_seconds_cross'))}s</td>"
                f"<td>{_f(s.get('mean_peak_memory_mb_cross'))}MB</td>"
                f"<td>{_f(s.get('mean_ground_truth_error'))}</td>"
                f"<td>{s.get('successful_runs',0)}/{s.get('total_runs',0)}</td></tr>\n")

    # 失败统计
    fc_all: dict[str, int] = {}
    for s in within_seed:
        fbc = s.get("failure_count_by_type")
        if fbc:
            for k, v in fbc.items():
                fc_all[k] = fc_all.get(k, 0) + v
    fail_html = ("<p style='color:#22c55e'>✔ 无失败记录</p>" if not fc_all
                 else "<ul>" + "".join(
                     f"<li><span class='ft'>{k}</span>"
                     f" <span class='fc'>{v}</span></li>"
                     for k, v in sorted(fc_all.items(), key=lambda x: -x[1]))
                 + "</ul>")

    # 失效边界图片
    fb_html = ""
    for fn, lbl in [("failure_boundary_accuracy.png", "精度"),
                     ("failure_boundary_runtime.png", "运行时间"),
                     ("failure_boundary_memory.png", "内存"),
                     ("failure_boundary_success.png", "成功率")]:
        fp = os.path.join(fb_dir or os.path.join(output_dir, "failure_boundary"), fn)
        if os.path.isfile(fp):
            fb_html += (f'<div class="fb-card"><div class="fb-lbl">{lbl}</div>'
                        f'<img src="failure_boundary/{fn}" alt="{lbl}"></div>\n')

    # 环境网格
    env_items = [
        ("操作系统", f"{env_info.get('os','?')} {env_info.get('os_release','')}"),
        ("Python", (env_info.get('python_version', '?') or '').split('\\n')[0]),
        ("CPU", f"{env_info.get('cpu_logical_cores','?')} 逻辑核心"),
        ("内存", f"{env_info.get('total_ram_gb','?')} GB"),
        ("GPU", env_info.get('gpu_name') or 'N/A'),
        ("显存", f"{env_info.get('gpu_vram_mb')} MB" if env_info.get('gpu_vram_mb') else 'N/A'),
        ("CUDA", env_info.get('cuda_version') or 'N/A'),
        ("Git", (env_info.get('git_commit') or 'N/A')[:8] if env_info.get('git_commit') else 'N/A'),
        ("Config SHA", (env_info.get('config_sha256') or 'N/A')[:16] + '...'),
    ]
    env_grid = "".join(
        f'<div class="ei"><div class="ek">{k}</div><div class="ev">{v}</div></div>'
        for k, v in env_items)

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>OTBench Benchmark v1.0 — 实验报告</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.6}}
h1{{color:#6366f1;font-size:26px}}h2{{margin:24px 0 12px;border-bottom:2px solid #1e293b;padding-bottom:8px;font-size:17px}}
.sub{{color:#64748b;font-size:13px;margin-bottom:20px}}
.card{{background:#1e293b;border-radius:12px;padding:20px;margin:14px 0}}
.metrics{{display:flex;gap:16px;flex-wrap:wrap}}
.metric{{background:#0f172a;border-radius:8px;padding:18px;flex:1;min-width:140px;text-align:center}}
.metric .v{{font-size:28px;font-weight:bold;color:#6366f1}}
.metric .l{{font-size:12px;color:#64748b;margin-top:6px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#0f172a;padding:8px 10px;text-align:left;color:#94a3b8;font-size:11px;text-transform:uppercase}}
td{{padding:6px 10px;border-bottom:1px solid #1e293b}}
tr.gh td{{background:#334155;font-size:12px;padding:6px 10px;border-bottom:2px solid #475569}}
tr:hover td{{background:rgba(99,102,241,.06)}}tr.gh:hover td{{background:#334155}}
.env{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}}
.ei{{padding:10px 14px;background:#0f172a;border-radius:6px}}
.ek{{font-size:10px;color:#64748b;text-transform:uppercase}}
.ev{{font-size:14px;margin-top:2px}}
.fb{{display:flex;flex-wrap:wrap;gap:12px}}
.fb-card{{flex:1;min-width:300px;max-width:49%}}
.fb-card img{{width:100%;border-radius:8px;border:1px solid #334155}}
.fb-lbl{{font-size:13px;color:#94a3b8;margin-bottom:4px}}
a{{color:#6366f1;text-decoration:none}}a:hover{{text-decoration:underline}}
.warn{{background:#422006;border:1px solid #f59e0b;border-radius:8px;padding:14px 18px;margin:16px 0;font-size:13px;color:#fbbf24}}
.warn::before{{content:'\\26A0\\FE0F ';font-size:16px}}
.ft{{color:#ef4444}}.fc{{font-weight:bold;color:#f59e0b;margin-left:8px}}
.dim{{color:#64748b;font-weight:normal}}
.footer{{text-align:center;color:#64748b;font-size:11px;margin-top:40px;padding-top:20px;border-top:1px solid #1e293b}}
ul{{padding-left:20px}}li{{margin:4px 0}}
</style></head><body>
<h1>OTBench Benchmark v1.0</h1>
<p class="sub">实验报告 · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · 华南理工大学</p>
<div class="metrics">
<div class="metric"><div class="v">{n_scenarios}</div><div class="l">场景数</div></div>
<div class="metric"><div class="v">{n_algorithms}</div><div class="l">算法数</div></div>
<div class="metric"><div class="v">{executed}</div><div class="l">执行次数</div></div>
<div class="metric"><div class="v">{rate:.1%}</div><div class="l">成功率</div></div>
<div class="metric"><div class="v">{n_runs}</div><div class="l">重复/Seed</div></div>
<div class="metric"><div class="v">{total_planned}</div><div class="l">计划总数</div></div>
</div>
<div class="warn">
<b>关于评分与参考类型</b><br>
<code>ground_truth_error</code> 仅在有 EMD 精确解时有效。
<code>reference_error</code> 相对高精度 Sinkhorn 参考解 — <b>这是参考解，不是精确真值</b>。
<code>consensus_deviation</code> 是辅助诊断指标，表示偏离候选算法中位数的程度。
综合评分受权重和参考类型影响，不代表算法绝对优劣。
不同 <code>ranking_group</code> 的算法不可直接对比排名。
</div>
<h2>实验环境</h2>
<div class="card"><div class="env">{env_grid}</div></div>
<h2>Within-Seed 结果</h2>
<div class="card"><table>
<tr><th>场景</th><th>算法</th><th>Seed</th><th>成功率</th><th>耗时</th><th>内存</th><th>GT Error</th><th>成功/总数</th></tr>
{algo_html}</table></div>
<h2>Cross-Seed 汇总</h2>
<div class="card"><table>
<tr><th>场景</th><th>算法</th><th>Seed 数</th><th>成功率</th><th>耗时</th><th>内存</th><th>GT Error</th><th>成功/总数</th></tr>
{cs_html if cs_html else '<tr><td colspan="8" style="color:#64748b">无跨 seed 数据</td></tr>'}</table></div>
<h2>失败统计</h2><div class="card">{fail_html}</div>
<h2>失效边界</h2><div class="card">{f'<div class="fb">{fb_html}</div>' if fb_html else '<p style="color:#64748b">无数据 (--no-sweep / quick 模式)</p>'}</div>
<h2>输出文件</h2><div class="card"><ul>
<li><a href="raw_results.csv">raw_results.csv</a> — 原始记录</li>
<li><a href="raw_results.jsonl">raw_results.jsonl</a> — 原始记录 (JSONL)</li>
<li><a href="summary_results.csv">summary_results.csv</a> — within-seed 统计</li>
<li><a href="summary_cross_seed.csv">summary_cross_seed.csv</a> — cross-seed 统计</li>
<li><a href="summary_results.json">summary_results.json</a> — 统计 (JSON)</li>
<li><a href="environment.json">environment.json</a> · <a href="protocol.md">protocol.md</a> · <a href="run_manifest.json">run_manifest.json</a></li>
<li><a href="failure_boundary/">failure_boundary/</a> — 失效边界图表</li>
</ul></div>
<div class="footer">OTBench Benchmark v1.0 · 华南理工大学 计算机科学与工程学院 · 大学生创新创业训练计划项目</div>
</body></html>"""
    hp = os.path.join(output_dir, "report.html")
    with open(hp, "w", encoding="utf-8") as f:
        f.write(html)
    return hp


def _f(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{v:.4g}"
    return str(v)
