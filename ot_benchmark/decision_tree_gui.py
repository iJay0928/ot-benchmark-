#!/usr/bin/env python3
"""Visual desktop application for the OT decision-tree selector."""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ot_benchmark"))

from algorithm_selector import (
    FEATURE_NAMES,
    collect_training_records,
    extract_lightweight_features,
    generate_training_datasets,
    load_selector_model,
    predict_algorithm_from_features,
    save_selector_model,
    save_training_records,
    select_training_algorithms,
    selector_weight_profiles,
    train_decision_tree,
)
from algorithms.registry import ALGORITHMS
from backend.base import get_backend
from downstream_knn import (
    KNN_OT_ADAPTERS,
    generate_domain_shift_dataset,
    run_knn_downstream_validation,
    save_knn_validation_results,
    plot_knn_validation_results,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "selector_results" / "ot_algorithm_selector.joblib"
BG = "#0f172a"
PANEL = "#1e293b"
PANEL_2 = "#111827"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"
GREEN = "#22c55e"
YELLOW = "#facc15"


class SelectorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OT Benchmark - 算法选择决策树")
        self.root.geometry("1180x760")
        self.root.minsize(980, 650)
        self.root.configure(bg=BG)
        self.bundle = None
        self.current_dataset = None
        self.tree_canvas = None
        self._configure_style()
        self._build_header()
        self._build_tabs()
        if DEFAULT_MODEL.exists():
            self._load_model(DEFAULT_MODEL, quiet=True)

    def _configure_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab", background=PANEL, foreground=MUTED,
            padding=(18, 9), font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map("TNotebook.Tab", background=[("selected", "#0c4a6e")],
                  foreground=[("selected", "white")])
        style.configure("Treeview", background=PANEL_2, foreground=TEXT,
                        fieldbackground=PANEL_2, rowheight=27, borderwidth=0)
        style.configure("Treeview.Heading", background="#334155", foreground=TEXT,
                        font=("Microsoft YaHei UI", 9, "bold"))

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(20, 12))
        tk.Label(header, text="OT 算法智能选择器", bg=BG, fg=TEXT,
                 font=("Microsoft YaHei UI", 22, "bold")).pack(anchor="w")
        tk.Label(
            header,
            text="输入数据集特征，决策树自动推荐适合的最优传输算法",
            bg=BG, fg=MUTED, font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", pady=(4, 0))
        self.model_status = tk.Label(header, text="尚未加载模型", bg=BG, fg=YELLOW,
                                     font=("Microsoft YaHei UI", 9))
        self.model_status.pack(anchor="e")

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=24, pady=(0, 22))
        self.predict_tab = tk.Frame(self.notebook, bg=BG)
        self.tree_tab = tk.Frame(self.notebook, bg=BG)
        self.knn_tab = tk.Frame(self.notebook, bg=BG)
        self.train_tab = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.predict_tab, text="  一键推荐  ")
        self.notebook.add(self.tree_tab, text="  决策树结构  ")
        self.notebook.add(self.knn_tab, text="  KNN 下游验证  ")
        self.notebook.add(self.train_tab, text="  训练新模型  ")
        self._build_predict_tab()
        self._build_tree_tab()
        self._build_knn_tab()
        self._build_train_tab()

    def _panel(self, parent):
        return tk.Frame(parent, bg=PANEL, highlightthickness=1,
                        highlightbackground="#334155")

    def _field(self, parent, label, variable, row, choices=None):
        tk.Label(parent, text=label, bg=PANEL, fg=MUTED,
                 font=("Microsoft YaHei UI", 9)).grid(row=row, column=0, sticky="w", padx=16, pady=9)
        if choices:
            widget = ttk.Combobox(parent, textvariable=variable, values=choices, state="readonly")
        else:
            widget = tk.Entry(parent, textvariable=variable, bg=PANEL_2, fg=TEXT,
                              insertbackground=TEXT, relief="flat", width=22)
        widget.grid(row=row, column=1, sticky="ew", padx=16, pady=9)
        return widget

    def _build_predict_tab(self):
        body = tk.Frame(self.predict_tab, bg=BG)
        body.pack(fill="both", expand=True, pady=12)
        left = self._panel(body)
        right = self._panel(body)
        left.pack(side="left", fill="y", padx=(0, 10))
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        tk.Label(left, text="数据集参数", bg=PANEL, fg=TEXT,
                 font=("Microsoft YaHei UI", 14, "bold")).grid(
                     row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 8))
        self.dimension = tk.StringVar(value="500")
        self.sparsity = tk.StringVar(value="0.85")
        self.noise = tk.StringVar(value="0.03")
        self.source_type = tk.StringVar(value="zipf")
        self.target_type = tk.StringVar(value="pareto")
        self.geometry_dim = tk.StringVar(value="1")
        self.structure_type = tk.StringVar(value="random")
        self.preference = tk.StringVar(value="balanced")
        self.mass_mode = tk.StringVar(value="balanced")
        self.seed = tk.StringVar(value="42")
        distributions = ["uniform", "zipf", "pareto", "gaussian_mixture"]
        self._field(left, "维度", self.dimension, 1)
        self._field(left, "稀疏度（0～0.99）", self.sparsity, 2)
        self._field(left, "噪声水平（0～1）", self.noise, 3)
        self._field(left, "源分布", self.source_type, 4, distributions)
        self._field(left, "目标分布", self.target_type, 5, distributions)
        self._field(left, "支撑点几何维度", self.geometry_dim, 6, ["1", "2", "5", "10", "50"])
        self._field(left, "稀疏结构", self.structure_type, 7,
                    ["random", "block", "group", "hierarchical"])
        self._field(left, "优化目标", self.preference, 8,
                    ["balanced", "accuracy", "speed", "memory"])
        self._field(left, "质量约束", self.mass_mode, 9, ["balanced", "unbalanced"])
        self._field(left, "随机种子", self.seed, 10)
        left.columnconfigure(1, weight=1)
        self.predict_button = tk.Button(
            left, text="生成数据并推荐算法", command=self.predict,
            bg=ACCENT, fg="#082f49", activebackground="#7dd3fc",
            relief="flat", padx=22, pady=11,
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        self.predict_button.grid(row=11, column=0, columnspan=2, padx=16, pady=(18, 10), sticky="ew")
        tk.Button(
            left, text="选择其他模型…", command=self.choose_model,
            bg="#334155", fg=TEXT, activebackground="#475569",
            relief="flat", padx=20, pady=8,
            font=("Microsoft YaHei UI", 9),
        ).grid(row=12, column=0, columnspan=2, padx=16, pady=(0, 16), sticky="ew")

        tk.Label(right, text="推荐结果", bg=PANEL, fg=TEXT,
                 font=("Microsoft YaHei UI", 14, "bold")).pack(anchor="w", padx=20, pady=(18, 6))
        self.recommendation = tk.Label(right, text="等待输入数据…", bg=PANEL, fg=MUTED,
                                       font=("Microsoft YaHei UI", 24, "bold"))
        self.recommendation.pack(anchor="w", padx=20, pady=(18, 2))
        self.confidence = tk.Label(right, text="", bg=PANEL, fg=GREEN,
                                   font=("Microsoft YaHei UI", 12, "bold"))
        self.confidence.pack(anchor="w", padx=20, pady=(0, 18))
        self.explanation = tk.Label(right, text="决策依据会显示在这里。", bg=PANEL, fg=MUTED,
                                    justify="left", anchor="w", wraplength=690,
                                    font=("Microsoft YaHei UI", 10))
        self.explanation.pack(fill="x", padx=20, pady=(0, 14))
        self.feature_table = ttk.Treeview(right, columns=("feature", "value"), show="headings")
        self.feature_table.heading("feature", text="数据特征")
        self.feature_table.heading("value", text="数值")
        self.feature_table.column("feature", width=260)
        self.feature_table.column("value", width=130, anchor="e")
        self.feature_table.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def _build_tree_tab(self):
        toolbar = tk.Frame(self.tree_tab, bg=BG)
        toolbar.pack(fill="x", pady=(12, 6))
        tk.Label(toolbar, text="绿色叶节点显示最终推荐算法；节点内是判断条件和样本数量。",
                 bg=BG, fg=MUTED, font=("Microsoft YaHei UI", 9)).pack(side="left")
        tk.Button(toolbar, text="刷新树图", command=self.draw_tree, bg="#334155", fg=TEXT,
                  relief="flat", padx=14, pady=6).pack(side="right")
        self.tree_host = self._panel(self.tree_tab)
        self.tree_host.pack(fill="both", expand=True)

    def _build_knn_tab(self):
        panel = self._panel(self.knn_tab)
        panel.pack(fill="both", expand=True, pady=12)
        header = tk.Frame(panel, bg=PANEL)
        header.pack(fill="x", padx=20, pady=(18, 8))
        tk.Label(header, text="OT 对齐后的 KNN 分类验证", bg=PANEL, fg=TEXT,
                 font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w")
        tk.Label(
            header,
            text="目标域标签只用于最终评价，不参与 OT 对齐或 KNN 训练。",
            bg=PANEL, fg=MUTED, font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(4, 0))

        controls = tk.Frame(panel, bg=PANEL)
        controls.pack(fill="x", padx=4, pady=4)
        self.knn_samples = tk.StringVar(value="240")
        self.knn_features = tk.StringVar(value="10")
        self.knn_classes = tk.StringVar(value="3")
        self.knn_shift = tk.StringVar(value="1.0")
        self.knn_noise = tk.StringVar(value="0.35")
        self.knn_neighbors = tk.StringVar(value="5")
        fields = [
            ("每个域样本数", self.knn_samples), ("特征维度", self.knn_features),
            ("类别数", self.knn_classes), ("域偏移强度", self.knn_shift),
            ("噪声", self.knn_noise), ("KNN 邻居数", self.knn_neighbors),
        ]
        for index, (label, variable) in enumerate(fields):
            cell = tk.Frame(controls, bg=PANEL)
            cell.grid(row=0, column=index, padx=8, pady=5, sticky="ew")
            tk.Label(cell, text=label, bg=PANEL, fg=MUTED,
                     font=("Microsoft YaHei UI", 8)).pack(anchor="w")
            tk.Entry(cell, textvariable=variable, width=11, bg=PANEL_2, fg=TEXT,
                     insertbackground=TEXT, relief="flat").pack(fill="x", pady=(3, 0), ipady=4)
            controls.columnconfigure(index, weight=1)

        algo_frame = tk.Frame(panel, bg=PANEL)
        algo_frame.pack(fill="x", padx=20, pady=7)
        tk.Label(algo_frame, text="参与验证的 OT 方法：", bg=PANEL, fg=MUTED,
                 font=("Microsoft YaHei UI", 9)).pack(side="left")
        self.knn_algorithm_vars = {}
        for name in KNN_OT_ADAPTERS:
            variable = tk.BooleanVar(value=name != "EMD exact")
            self.knn_algorithm_vars[name] = variable
            tk.Checkbutton(
                algo_frame, text=name, variable=variable, bg=PANEL, fg=TEXT,
                selectcolor=BG, activebackground=PANEL, activeforeground=TEXT,
                font=("Microsoft YaHei UI", 8),
            ).pack(side="left", padx=5)

        action_row = tk.Frame(panel, bg=PANEL)
        action_row.pack(fill="x", padx=20, pady=8)
        self.knn_run_button = tk.Button(
            action_row, text="运行 KNN 下游验证", command=self.start_knn_validation,
            bg=ACCENT, fg="#082f49", relief="flat", padx=24, pady=9,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.knn_run_button.pack(side="left")
        self.knn_status = tk.Label(action_row, text="等待运行", bg=PANEL, fg=MUTED,
                                   font=("Microsoft YaHei UI", 9))
        self.knn_status.pack(side="left", padx=16)

        columns = ("algorithm", "accuracy", "f1", "gain", "time", "memory", "status")
        self.knn_table = ttk.Treeview(panel, columns=columns, show="headings", height=12)
        headings = {
            "algorithm": "方法", "accuracy": "Accuracy", "f1": "Macro-F1",
            "gain": "准确率提升", "time": "OT+KNN耗时", "memory": "峰值内存", "status": "状态",
        }
        widths = {"algorithm": 180, "accuracy": 90, "f1": 90, "gain": 105,
                  "time": 105, "memory": 100, "status": 130}
        for column in columns:
            self.knn_table.heading(column, text=headings[column])
            self.knn_table.column(column, width=widths[column], anchor="center")
        self.knn_table.pack(fill="both", expand=True, padx=20, pady=(5, 18))

    def start_knn_validation(self):
        try:
            config = {
                "n_samples": int(self.knn_samples.get()),
                "n_features": int(self.knn_features.get()),
                "n_classes": int(self.knn_classes.get()),
                "shift_intensity": float(self.knn_shift.get()),
                "noise": float(self.knn_noise.get()),
                "neighbors": int(self.knn_neighbors.get()),
            }
            selected = [name for name, var in self.knn_algorithm_vars.items() if var.get()]
            if not selected:
                raise ValueError("请至少选择一种 OT 方法")
            if config["n_samples"] > 1000:
                raise ValueError("图形界面单次验证最多使用 1000 个样本，以避免耦合矩阵过大")
        except Exception as exc:
            messagebox.showerror("参数无效", str(exc))
            return
        for item in self.knn_table.get_children():
            self.knn_table.delete(item)
        self.knn_run_button.configure(state="disabled")
        self.knn_status.configure(text="正在生成领域偏移数据并运行 OT…", fg=YELLOW)
        threading.Thread(
            target=self._knn_worker, args=(config, selected), daemon=True
        ).start()

    def _knn_worker(self, config, selected):
        try:
            neighbors = config.pop("neighbors")
            dataset = generate_domain_shift_dataset(**config)
            results = run_knn_downstream_validation(dataset, selected, neighbors)
            self.root.after(0, lambda: self._knn_finished(dataset, results))
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self._knn_failed(error))

    def _knn_finished(self, dataset, results):
        self.knn_run_button.configure(state="normal")
        valid = [result for result in results if result.success]
        best = max(valid, key=lambda result: result.accuracy)
        csv_path, _ = save_knn_validation_results(
            ROOT / "selector_results", dataset, results
        )
        chart_path = plot_knn_validation_results(
            ROOT / "selector_results" / "knn_downstream_comparison.png", results
        )
        self.knn_status.configure(
            text=(f"完成 · 最佳：{best.algorithm} ({best.accuracy:.1%}) · "
                  f"已保存 {chart_path.name}"), fg=GREEN
        )
        for result in results:
            gain = f"{result.accuracy_gain:+.1%}" if result.success else "-"
            self.knn_table.insert("", "end", values=(
                result.algorithm,
                f"{result.accuracy:.1%}" if result.success else "-",
                f"{result.macro_f1:.1%}" if result.success else "-",
                gain,
                f"{result.runtime_seconds:.3f}s",
                f"{result.peak_memory_mb:.1f}MB",
                "成功" if result.success else f"失败: {result.error[:24]}",
            ))

    def _knn_failed(self, error):
        self.knn_run_button.configure(state="normal")
        self.knn_status.configure(text="验证失败", fg="#f87171")
        messagebox.showerror("KNN 下游验证失败", error)

    def _build_train_tab(self):
        panel = self._panel(self.train_tab)
        panel.pack(fill="both", expand=True, pady=12)
        tk.Label(panel, text="重新训练算法选择模型", bg=PANEL, fg=TEXT,
                 font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w", padx=22, pady=(20, 5))
        tk.Label(panel, text="系统会先评测候选 OT 算法，再用冠军标签训练决策树。训练期间窗口可以继续移动。",
                 bg=PANEL, fg=MUTED, font=("Microsoft YaHei UI", 10)).pack(anchor="w", padx=22)
        form = tk.Frame(panel, bg=PANEL)
        form.pack(anchor="w", padx=8, pady=18)
        self.train_samples = tk.StringVar(value="60")
        self.train_depth = tk.StringVar(value="5")
        self.train_leaf = tk.StringVar(value="2")
        self.train_seed = tk.StringVar(value="42")
        self._field(form, "训练任务数量", self.train_samples, 0)
        self._field(form, "最大树深", self.train_depth, 1)
        self._field(form, "叶节点最少样本", self.train_leaf, 2)
        self._field(form, "随机种子", self.train_seed, 3)
        self.train_button = tk.Button(
            panel, text="开始训练", command=self.start_training,
            bg=ACCENT, fg="#082f49", relief="flat", padx=28, pady=10,
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        self.train_button.pack(anchor="w", padx=22)
        self.train_status = tk.Label(panel, text="建议演示使用 60 条，论文实验使用 150 条以上。",
                                     bg=PANEL, fg=MUTED, justify="left",
                                     font=("Microsoft YaHei UI", 10))
        self.train_status.pack(anchor="w", padx=22, pady=16)

    def choose_model(self):
        path = filedialog.askopenfilename(
            title="选择决策树模型", initialdir=str(ROOT / "selector_results"),
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        )
        if path:
            self._load_model(Path(path))

    def _load_model(self, path: Path, quiet=False):
        try:
            self.bundle = load_selector_model(path)
            evaluation = self.bundle.get("evaluation", {})
            accuracy = evaluation.get("accuracy")
            suffix = f" · 测试准确率 {accuracy:.1%}" if isinstance(accuracy, (int, float)) else ""
            self.model_status.configure(text=f"已加载：{path.name}{suffix}", fg=GREEN)
            self.draw_tree()
        except Exception as exc:
            self.bundle = None
            self.model_status.configure(text="模型加载失败", fg="#f87171")
            if not quiet:
                messagebox.showerror("模型加载失败", str(exc))

    def predict(self):
        if self.bundle is None:
            messagebox.showwarning("需要模型", "请先加载或训练一个决策树模型。")
            return
        try:
            dimension = int(self.dimension.get())
            sparsity = float(self.sparsity.get())
            noise = float(self.noise.get())
            seed = int(self.seed.get())
            geometry_dim = int(self.geometry_dim.get())
            if dimension < 10 or not 0 <= sparsity < 1 or not 0 <= noise <= 1:
                raise ValueError("维度至少为 10，稀疏度范围为 [0,1)，噪声范围为 [0,1]。")
            if dimension > 200_000:
                raise ValueError("为保证交互速度，当前界面最多支持 200,000 维的轻量预测。")
            if self.mass_mode.get() == "unbalanced":
                self.recommendation.configure(text="Unbalanced OT", fg=ACCENT)
                self.confidence.configure(text="任务类型规则：质量不守恒")
                self.explanation.configure(
                    text=("源分布与目标分布总质量不相等时，普通平衡 OT 的约束不适用。"
                          "系统先通过任务类型分支选择 Unbalanced OT，不与平衡 OT 算法的距离直接混排。")
                )
                return
            params = (
                dimension, sparsity, noise, self.source_type.get(), self.target_type.get(),
                seed, geometry_dim, self.structure_type.get(), self.preference.get(),
            )
            self.predict_button.configure(state="disabled", text="正在分析数据特征…")
            self.recommendation.configure(text="计算中…", fg=YELLOW)
            self.confidence.configure(text="采用 O(n) 内存的高维轻量预测")
            threading.Thread(target=self._prediction_worker, args=(params,), daemon=True).start()
        except Exception as exc:
            messagebox.showerror("无法完成推荐", str(exc))

    def _prediction_worker(self, params):
        try:
            dimension, sparsity, noise, source_type, target_type, seed, geometry_dim, structure, preference = params
            profile_map = dict(selector_weight_profiles())
            selected_weights = profile_map[preference]
            features = extract_lightweight_features(
                dimension=dimension, sparsity=sparsity, noise_level=noise,
                source_dist_type=source_type, target_dist_type=target_type,
                random_state=seed, geometry_dimension=geometry_dim, structure_type=structure,
                preference_weights=(
                    selected_weights.w_accuracy, selected_weights.w_speed,
                    selected_weights.w_memory, selected_weights.w_success,
                ),
            )
            algorithm, confidence = predict_algorithm_from_features(features, self.bundle)
            self.root.after(
                0, lambda: self._prediction_finished(algorithm, confidence, features, dimension)
            )
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self._prediction_failed(error))

    def _prediction_finished(self, algorithm, confidence, features, dimension):
        self.predict_button.configure(state="normal", text="生成数据并推荐算法")
        self.recommendation.configure(text=algorithm, fg=ACCENT)
        self.confidence.configure(text=f"模型置信度：{confidence:.1%} · {dimension:,} 维轻量预测")
        important = self.bundle.get("evaluation", {}).get("feature_importance", [])[:3]
        names = "、".join(name for name, value in important if value > 0) or "当前决策路径"
        self.explanation.configure(
            text=(f"推荐依据：模型根据 {names} 等特征完成分类。代价矩阵统计量采用固定样本估算，"
                  "没有创建 n×n 矩阵；置信度表示叶节点训练样本的一致程度，不等同于算法准确率。")
        )
        for item in self.feature_table.get_children():
            self.feature_table.delete(item)
        for name in FEATURE_NAMES:
            self.feature_table.insert("", "end", values=(name, f"{features[name]:.6f}"))

    def _prediction_failed(self, error):
        self.predict_button.configure(state="normal", text="生成数据并推荐算法")
        self.recommendation.configure(text="预测失败", fg="#f87171")
        self.confidence.configure(text="")
        messagebox.showerror("无法完成推荐", error)

    def draw_tree(self):
        for child in self.tree_host.winfo_children():
            child.destroy()
        if self.bundle is None:
            tk.Label(self.tree_host, text="加载模型后将在这里显示决策树。",
                     bg=PANEL, fg=MUTED, font=("Microsoft YaHei UI", 11)).pack(expand=True)
            return
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            from sklearn.tree import plot_tree

            model = self.bundle["model"]
            figure = Figure(figsize=(12, 6), dpi=100, facecolor="#f8fafc")
            axis = figure.add_subplot(111)
            plot_tree(
                model, feature_names=FEATURE_NAMES, class_names=[str(x) for x in model.classes_],
                filled=True, rounded=True, impurity=False, proportion=True,
                fontsize=8, ax=axis,
            )
            axis.set_title("OT Algorithm Selection Decision Tree", fontsize=13, pad=12)
            figure.tight_layout()
            self.tree_canvas = FigureCanvasTkAgg(figure, master=self.tree_host)
            self.tree_canvas.draw()
            self.tree_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        except Exception as exc:
            tk.Label(self.tree_host, text=f"树图绘制失败：{exc}", bg=PANEL, fg="#f87171").pack(expand=True)

    def start_training(self):
        try:
            config = {
                "samples": int(self.train_samples.get()),
                "max_depth": int(self.train_depth.get()),
                "min_leaf": int(self.train_leaf.get()),
                "random_state": int(self.train_seed.get()),
            }
            if config["samples"] < 10:
                raise ValueError("训练任务数量至少为 10。")
        except Exception as exc:
            messagebox.showerror("训练参数无效", str(exc))
            return
        self.train_button.configure(state="disabled")
        self.train_status.configure(text="正在生成数据并评测候选算法，请稍候…", fg=YELLOW)
        threading.Thread(target=self._training_worker, args=(config,), daemon=True).start()

    def _training_worker(self, config):
        try:
            output = ROOT / "selector_results"
            algorithms = select_training_algorithms(ALGORITHMS)
            datasets = generate_training_datasets(config["samples"], config["random_state"])
            records = collect_training_records(
                datasets, algorithms, backend=get_backend("numpy"), progress=False,
                weight_profiles=selector_weight_profiles(),
            )
            model, evaluation = train_decision_tree(
                records, max_depth=config["max_depth"],
                min_samples_leaf=config["min_leaf"], random_state=config["random_state"],
            )
            output.mkdir(parents=True, exist_ok=True)
            save_training_records(output / "selector_training_records.csv", records)
            model_path = save_selector_model(
                output / "ot_algorithm_selector.joblib", model, evaluation,
                config | {"algorithms": list(algorithms), "valid_records": len(records)},
            )
            self.root.after(0, lambda: self._training_finished(model_path, evaluation))
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self._training_failed(error))

    def _training_finished(self, path, evaluation):
        self.train_button.configure(state="normal")
        classes = "，".join(f"{k}: {v}" for k, v in evaluation.class_counts.items())
        self.train_status.configure(
            text=(f"训练完成：测试准确率 {evaluation.accuracy:.1%}，平衡准确率 "
                  f"{evaluation.balanced_accuracy:.1%}\n标签分布：{classes}"), fg=GREEN,
        )
        self._load_model(path, quiet=True)
        self.notebook.select(self.tree_tab)
        messagebox.showinfo("训练完成", "新模型已保存并自动加载。")

    def _training_failed(self, error):
        self.train_button.configure(state="normal")
        self.train_status.configure(text=f"训练失败：{error}", fg="#f87171")
        messagebox.showerror("训练失败", error)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    SelectorApp().run()
