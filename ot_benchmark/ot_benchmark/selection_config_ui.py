#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法与数据集选择 GUI
====================
在正式评测前选择要运行的算法和数据集。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import Any


BG = "#0f172a"
PANEL = "#1e293b"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"


DATASET_SCENARIOS = {
    "ds1-baseline": "基准分布对齐：模拟两个相似人群、商品库或传感器批次之间的基础差异。",
    "ds2-recommendation": "推荐系统：模拟用户只评分少量电影/商品时，两个用户兴趣分布之间的差异。",
    "ds3-textclassification": "文本分类/搜索：模拟长文本词袋中少量关键词活跃时，不同文档主题分布的迁移。",
    "ds4-bioinformatics": "生物信息学：模拟基因表达或细胞亚群数据中，多簇信号之间的分布差异。",
    "ds5-stresstest": "超大规模压力测试：模拟极高维日志、广告点击或商品目录中的稀疏行为数据。",
    "ds6-structuredsparse": "结构化稀疏：模拟用户群组、商品类目或模块化传感器里成块出现的稀疏信号。",
    "ds7-extremesparse": "大规模推荐/广告：模拟海量物品中用户只交互极少数对象的极端稀疏场景。",
    "ds8-highnoise": "高噪声鲁棒性：模拟采集误差、异常点击或低质量标注较多的数据。",
    "quick-ds1": "快速冒烟测试：模拟小规模、较稳定的日常分布对齐任务。",
    "quick-ds2": "快速高稀疏测试：模拟点击、评分或词频这类少数位置有信号的数据。",
    "shift-ds1-covariate": "训练/上线环境变化：模拟用户画像、设备环境或地区结构变了，但任务含义基本不变。",
    "shift-ds2-label": "类别占比变化：模拟节假日商品销量、热门主题或疾病类型占比突然改变。",
    "shift-ds3-conceptdrift": "概念漂移：模拟用户偏好、文本含义或业务规则随时间改变。",
    "struct-ds1-block": "块结构行为：模拟同一商品类目、社区或设备组同时活跃的块状稀疏数据。",
    "struct-ds2-group": "分组结构行为：模拟用户群体、兴趣标签组或业务分组之间的迁移。",
    "struct-ds3-hierarchical": "层级结构行为：模拟商品类目树、组织层级或生物通路的多层稀疏模式。",
    "struct-ds4-random": "随机稀疏基线：模拟没有明显业务结构的随机缺失或随机活跃数据。",
}


def describe_dataset_scenario(dataset: Any) -> str:
    """Return a user-facing scenario description for a benchmark dataset."""
    name = getattr(dataset, "name", "")
    name_key = name.lower()
    origin = str(getattr(dataset, "data_origin", "")).lower()

    for key, scenario in DATASET_SCENARIOS.items():
        if key in name_key:
            return scenario

    if "movielens" in name_key or "movielens" in origin:
        return "真实电影评分：模拟流媒体/电商推荐里，不同用户兴趣分布之间的距离。"
    if "uci-bow" in name_key or "bag of words" in origin:
        return "真实文本词频：模拟新闻、邮件或论文文档中，不同文章主题词分布的差异。"
    if "libsvm" in name_key or "libsvm" in origin:
        return "真实稀疏分类特征：模拟医学诊断、文本分类或广告点击样本的高维特征迁移。"
    if "scrna" in name_key or "scrna" in origin:
        return "单细胞基因表达：模拟不同细胞类型或实验批次之间的表达分布变化。"

    desc = str(getattr(dataset, "description", "")).lower()
    sparsity = float(getattr(dataset, "sparsity", 0.0) or 0.0)
    if "zipf" in desc:
        return "长尾行为数据：模拟少数热门项目占多数流量的点击、购买或词频分布。"
    if "pareto" in desc:
        return "重尾风险/消费数据：模拟少数极大值主导的消费额、访问量或异常事件。"
    if sparsity >= 0.9:
        return "高维稀疏行为：模拟用户只点击少量商品、文档只包含少量关键词的场景。"
    if sparsity >= 0.75:
        return "中高维稀疏信号：模拟推荐、文本或传感器数据里部分特征缺失/不活跃。"
    return "通用分布对齐：模拟两个样本群体、业务周期或数据批次之间的差异。"


def format_dataset_option(dataset: Any) -> str:
    scenario = describe_dataset_scenario(dataset)
    technical = getattr(dataset, "description", "")
    origin = getattr(dataset, "data_origin", "unknown")
    geometry = getattr(dataset, "geometry_description", "index position (1D)")
    return (
        f"场景: {scenario}\n技术特征: {technical} | 几何: {geometry} | 来源: {origin}"
    )


class ScrollableCheckList(tk.Frame):
    """带滚动条的复选框列表。"""

    def __init__(self, parent, title: str, items: list[tuple[str, str]], height: int = 420):
        super().__init__(parent, bg=PANEL)
        self.vars: dict[str, tk.BooleanVar] = {}

        header = tk.Frame(self, bg=PANEL)
        header.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(
            header, text=title, bg=PANEL, fg=TEXT,
            font=("Microsoft YaHei UI", 12, "bold")
        ).pack(side="left")

        actions = tk.Frame(header, bg=PANEL)
        actions.pack(side="right")
        for label, command in [
            ("全选", self.select_all),
            ("清空", self.clear_all),
            ("反选", self.invert),
        ]:
            tk.Button(
                actions, text=label, command=command,
                bg="#334155", fg=TEXT, activebackground="#475569",
                activeforeground=TEXT, relief="flat", padx=8, pady=2,
                font=("Microsoft YaHei UI", 9),
            ).pack(side="left", padx=3)

        outer = tk.Frame(self, bg=PANEL)
        outer.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0, height=height)
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self.inner = tk.Frame(canvas, bg=PANEL)

        self.inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for key, desc in items:
            var = tk.BooleanVar(value=True)
            self.vars[key] = var
            row = tk.Frame(self.inner, bg=PANEL)
            row.pack(fill="x", padx=4, pady=3)
            cb = tk.Checkbutton(
                row, text=key, variable=var,
                bg=PANEL, fg=TEXT, selectcolor=BG,
                activebackground=PANEL, activeforeground=TEXT,
                font=("Microsoft YaHei UI", 10),
                anchor="w",
            )
            cb.pack(fill="x", anchor="w")
            if desc:
                tk.Label(
                    row, text=desc, bg=PANEL, fg=MUTED,
                    font=("Microsoft YaHei UI", 8), anchor="w",
                    wraplength=420, justify="left",
                ).pack(fill="x", padx=(26, 0))

    def select_all(self):
        for var in self.vars.values():
            var.set(True)

    def clear_all(self):
        for var in self.vars.values():
            var.set(False)

    def invert(self):
        for var in self.vars.values():
            var.set(not var.get())

    def selected_keys(self) -> list[str]:
        return [key for key, var in self.vars.items() if var.get()]


class SelectionWindow:
    def __init__(self, algorithms: dict[str, dict[str, Any]], datasets: list[Any]):
        self.algorithms = algorithms
        self.datasets = datasets
        self.result: tuple[list[str], list[str]] | None = None

        self.root = tk.Tk()
        self.root.title("OT Benchmark - 选择评测算法与数据集")
        self.root.configure(bg=BG)
        self.root.geometry("1080x680")
        self.root.minsize(920, 560)

        title = tk.Label(
            self.root,
            text="选择本次要评测的算法和数据集",
            bg=BG, fg=TEXT,
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        title.pack(pady=(18, 4))

        subtitle = tk.Label(
            self.root,
            text="默认全选；取消勾选即可跳过对应算法或数据集。",
            bg=BG, fg=MUTED,
            font=("Microsoft YaHei UI", 10),
        )
        subtitle.pack(pady=(0, 14))

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=18)

        algo_items = [
            (name, info.get("description", ""))
            for name, info in algorithms.items()
        ]
        dataset_items = [
            (
                ds.name,
                format_dataset_option(ds),
            )
            for ds in datasets
        ]

        self.algo_list = ScrollableCheckList(body, "算法", algo_items)
        self.data_list = ScrollableCheckList(body, "数据集", dataset_items)
        self.algo_list.pack(side="left", fill="both", expand=True, padx=(0, 9))
        self.data_list.pack(side="left", fill="both", expand=True, padx=(9, 0))

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=18, pady=16)

        self.count_label = tk.Label(
            footer, bg=BG, fg=MUTED,
            font=("Microsoft YaHei UI", 10),
        )
        self.count_label.pack(side="left")
        self._refresh_count()

        for checklist in [self.algo_list, self.data_list]:
            for var in checklist.vars.values():
                var.trace_add("write", lambda *_args: self._refresh_count())

        tk.Button(
            footer, text="取消", command=self._cancel,
            bg="#334155", fg=TEXT, activebackground="#475569",
            activeforeground=TEXT, relief="flat", padx=18, pady=8,
            font=("Microsoft YaHei UI", 10),
        ).pack(side="right", padx=(8, 0))

        tk.Button(
            footer, text="开始评测", command=self._confirm,
            bg=ACCENT, fg="#082f49", activebackground="#7dd3fc",
            activeforeground="#082f49", relief="flat", padx=24, pady=8,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self._cancel)

    def _refresh_count(self):
        n_algos = len(self.algo_list.selected_keys())
        n_datasets = len(self.data_list.selected_keys())
        self.count_label.configure(text=f"已选择 {n_algos} 个算法 × {n_datasets} 个数据集")

    def _confirm(self):
        selected_algos = self.algo_list.selected_keys()
        selected_datasets = self.data_list.selected_keys()
        if not selected_algos:
            messagebox.showwarning("需要选择算法", "请至少选择一个算法。")
            return
        if not selected_datasets:
            messagebox.showwarning("需要选择数据集", "请至少选择一个数据集。")
            return
        self.result = (selected_algos, selected_datasets)
        self.root.destroy()

    def _cancel(self):
        self.result = None
        self.root.destroy()

    def run(self) -> tuple[list[str], list[str]] | None:
        self.root.mainloop()
        return self.result


def launch_selection_ui(
    algorithms: dict[str, dict[str, Any]],
    datasets: list[Any],
) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    """打开选择窗口并返回筛选后的算法/数据集。取消时返回空选择。"""
    try:
        window = SelectionWindow(algorithms, datasets)
        result = window.run()
    except tk.TclError as exc:
        print(f"\n图形选择界面不可用，切换到终端交互选择。原因: {exc}\n")
        result = launch_console_selection_ui(algorithms, datasets)

    if result is None:
        return {}, []

    selected_algos, selected_datasets = result
    selected_algo_set = set(selected_algos)
    selected_dataset_set = set(selected_datasets)

    filtered_algorithms = {
        name: info for name, info in algorithms.items()
        if name in selected_algo_set
    }
    filtered_datasets = [
        ds for ds in datasets
        if ds.name in selected_dataset_set
    ]
    return filtered_algorithms, filtered_datasets


def _parse_selection(raw_value: str, total: int) -> list[int] | None:
    """Parse 1-based selections like all, 1,3,5-8, or empty for all."""
    raw_value = raw_value.strip().lower()
    if raw_value in ("", "all", "a", "*"):
        return list(range(total))
    if raw_value in ("none", "n", "cancel", "q"):
        return None

    selected: set[int] = set()
    for part in raw_value.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            if start > end:
                start, end = end, start
            selected.update(range(start - 1, end))
        else:
            selected.add(int(part) - 1)

    if any(index < 0 or index >= total for index in selected):
        raise ValueError(f"选择范围必须在 1-{total} 之间")
    return sorted(selected)


def _console_choose(title: str, rows: list[tuple[str, str]]) -> list[str] | None:
    print(f"\n{title}")
    print("-" * 72)
    for index, (name, desc) in enumerate(rows, 1):
        suffix = f"  |  {desc}" if desc else ""
        print(f"{index:>2}. {name}{suffix}")

    while True:
        try:
            raw_value = input("请输入编号(如 1,3,5-8)，all=全选，q=取消，直接回车=全选: ")
        except EOFError:
            raw_value = "all"

        try:
            indices = _parse_selection(raw_value, len(rows))
        except (TypeError, ValueError) as exc:
            print(f"输入无效: {exc}")
            continue

        if indices is None:
            return None
        if not indices:
            print("至少选择一项。")
            continue
        return [rows[index][0] for index in indices]


def launch_console_selection_ui(
    algorithms: dict[str, dict[str, Any]],
    datasets: list[Any],
) -> tuple[list[str], list[str]] | None:
    """Terminal fallback used when Tk cannot create a window."""
    algo_rows = [
        (name, info.get("description", ""))
        for name, info in algorithms.items()
    ]
    dataset_rows = [
        (
            ds.name,
            format_dataset_option(ds),
        )
        for ds in datasets
    ]

    selected_algos = _console_choose("选择本次要评测的算法", algo_rows)
    if selected_algos is None:
        return None

    selected_datasets = _console_choose("选择本次要评测的数据集", dataset_rows)
    if selected_datasets is None:
        return None

    print(f"\n已选择 {len(selected_algos)} 个算法 × {len(selected_datasets)} 个数据集。\n")
    return selected_algos, selected_datasets


def launch_dimension_tier_ui(default: str = "standard") -> str:
    """Ask for the high-dimensional benchmark tier before datasets are loaded."""
    from .dimension_tiers import DIMENSION_TIERS

    try:
        root = tk.Tk()
        root.title("OT Benchmark - 高维评测档位")
        root.configure(bg=BG)
        root.geometry("620x420")
        root.resizable(False, False)

        selected = tk.StringVar(value=default if default in DIMENSION_TIERS else "standard")
        result = {"value": default}

        tk.Label(
            root, text="选择本次高维评测档位", bg=BG, fg=TEXT,
            font=("Microsoft YaHei UI", 17, "bold"),
        ).pack(pady=(22, 6))
        tk.Label(
            root, text="维度越高，数据准备和算法运行时间越长。", bg=BG, fg=MUTED,
            font=("Microsoft YaHei UI", 10),
        ).pack(pady=(0, 14))

        options = tk.Frame(root, bg=PANEL)
        options.pack(fill="both", expand=True, padx=28, pady=4)
        for key, info in DIMENSION_TIERS.items():
            row = tk.Frame(options, bg=PANEL)
            row.pack(fill="x", padx=16, pady=7)
            tk.Radiobutton(
                row,
                text=f"{info['label']} ({info['target_dim']:,} 维)",
                value=key,
                variable=selected,
                bg=PANEL,
                fg=TEXT,
                selectcolor=BG,
                activebackground=PANEL,
                activeforeground=TEXT,
                font=("Microsoft YaHei UI", 11, "bold"),
                anchor="w",
            ).pack(fill="x")
            tk.Label(
                row, text=info["description"], bg=PANEL, fg=MUTED,
                font=("Microsoft YaHei UI", 9), anchor="w",
            ).pack(fill="x", padx=(26, 0))

        def confirm():
            result["value"] = selected.get()
            root.destroy()

        tk.Button(
            root, text="确定", command=confirm, bg=ACCENT, fg="#082f49",
            activebackground="#7dd3fc", relief="flat", padx=30, pady=8,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(pady=18)
        root.protocol("WM_DELETE_WINDOW", confirm)
        root.mainloop()
        return result["value"]
    except tk.TclError:
        print("\n选择高维评测档位：")
        keys = list(DIMENSION_TIERS)
        for index, key in enumerate(keys, 1):
            info = DIMENSION_TIERS[key]
            print(f"  {index}. {info['label']} ({info['target_dim']:,} 维) - {info['description']}")
        try:
            raw = input(f"请输入编号，直接回车使用默认档位 {default}: ").strip()
            return keys[int(raw) - 1] if raw else default
        except (EOFError, ValueError, IndexError):
            return default
