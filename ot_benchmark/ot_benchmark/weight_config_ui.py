#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重配置 GUI
============
现代化的权重配置窗口 —— 拖拽滑块实时调整评测维度权重。
"""

import tkinter as tk
from tkinter import ttk
try:
    from .benchmark_engine import WeightConfig
except ImportError:
    from benchmark_engine import WeightConfig


PRESETS = {
    "平衡": (0.35, 0.30, 0.25, 0.10),
    "精度优先": (0.55, 0.20, 0.15, 0.10),
    "速度优先": (0.20, 0.50, 0.20, 0.10),
    "内存优先": (0.20, 0.20, 0.50, 0.10),
    "均等": (0.25, 0.25, 0.25, 0.25),
}

DIM_LABELS = ["精度", "速度", "内存", "成功率"]
DIM_KEYS = ["w_accuracy", "w_speed", "w_memory", "w_success"]
DIM_COLORS = ["#f43f5e", "#f97316", "#3b82f6", "#10b981"]


class WeightSlider(tk.Canvas):
    def __init__(self, parent, value=50, color="#3b82f6", command=None, **kwargs):
        super().__init__(parent, width=280, height=40, bg="#0f172a", highlightthickness=0, **kwargs)
        self._value = value
        self._color = color
        self._command = command
        self._dragging = False
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.draw()

    def draw(self):
        self.delete("all")
        trough_y = 20
        fill_width = (self._value / 100) * 280
        
        self.create_rectangle(0, trough_y - 3, 280, trough_y + 3, fill="#334155", outline="")
        if fill_width > 0:
            self.create_rectangle(0, trough_y - 3, fill_width, trough_y + 3, fill=self._color, outline="")
        
        knob_x = fill_width
        self.create_oval(knob_x - 11, trough_y - 11, knob_x + 11, trough_y + 11, 
                         fill=self._color, outline="#fff", width=2)

    def _on_click(self, event):
        self._dragging = True
        self._update_value(event.x)

    def _on_drag(self, event):
        if self._dragging:
            self._update_value(event.x)

    def _on_release(self, _event):
        self._dragging = False

    def _update_value(self, x):
        value = max(0, min(100, int((x / 280) * 100)))
        if value != self._value:
            self._value = value
            self.draw()
            if self._command:
                self._command(value)

    def get(self):
        return self._value

    def set(self, value):
        self._value = max(0, min(100, int(value)))
        self.draw()


class FlatButton(tk.Canvas):
    def __init__(self, parent, text, command=None, 
                 bg="#334155", hover_bg="#475569", 
                 fg="#e2e8f0", hover_fg="white",
                 font=("Segoe UI", 10), 
                 width=100, height=36, **kwargs):
        super().__init__(parent, width=width, height=height, 
                         bg="#0f172a", highlightthickness=0, **kwargs)
        self._text = text
        self._command = command
        self._bg = bg
        self._hover_bg = hover_bg
        self._fg = fg
        self._hover_fg = hover_fg
        self._font = font
        self._is_hover = False
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.draw()

    def draw(self):
        self.delete("all")
        bg_color = self._hover_bg if self._is_hover else self._bg
        fg_color = self._hover_fg if self._is_hover else self._fg
        
        radius = 8
        self.create_rounded_rect(0, 0, self.winfo_width(), self.winfo_height(), radius, fill=bg_color)
        self.create_text(self.winfo_width() // 2, self.winfo_height() // 2,
                         text=self._text, fill=fg_color, font=self._font, anchor="center")

    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1, x2, y1 + radius,
            x2, y2 - radius,
            x2, y2, x2 - radius, y2,
            x1 + radius, y2,
            x1, y2, x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_enter(self, _event):
        self._is_hover = True
        self.draw()

    def _on_leave(self, _event):
        self._is_hover = False
        self.draw()

    def _on_click(self, _event):
        if self._command:
            self._command()


class WeightConfigWindow:
    def __init__(self):
        self.result: WeightConfig | None = None

        self.root = tk.Tk()
        self.root.title("OT Benchmark")
        self.root.geometry("640x620")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f172a")

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - 640) // 2
        y = (sh - 620) // 2
        self.root.geometry(f"640x620+{x}+{y}")

        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#0f172a")
        header.pack(fill="x", padx=40, pady=(32, 8))

        title = tk.Label(header, text="评测权重配置", font=("Segoe UI", 22, "bold"),
                         fg="#f8fafc", bg="#0f172a")
        title.pack(anchor="w")
        
        subtitle = tk.Label(header, text="调整四维指标权重，权重将自动归一化", 
                            font=("Segoe UI", 11), fg="#64748b", bg="#0f172a")
        subtitle.pack(anchor="w", pady=(4, 0))

        preset_frame = tk.Frame(self.root, bg="#0f172a")
        preset_frame.pack(fill="x", padx=40, pady=(20, 0))

        self._preset_buttons = {}
        for i, name in enumerate(PRESETS.keys()):
            btn = FlatButton(
                preset_frame, text=name,
                bg="#1e293b", hover_bg="#334155",
                fg="#cbd5e1", hover_fg="white",
                font=("Segoe UI", 10),
                width=100, height=36,
                command=lambda n=name: self._apply_preset(n),
            )
            btn.pack(side="left", padx=4)
            self._preset_buttons[name] = btn

        slider_area = tk.Frame(self.root, bg="#0f172a")
        slider_area.pack(fill="x", padx=40, pady=(28, 0))

        self._sliders: list[WeightSlider] = []
        self._pct_labels: list[tk.Label] = []
        self._val_entries: list[tk.Entry] = []

        for i, (label, color) in enumerate(zip(DIM_LABELS, DIM_COLORS)):
            row = tk.Frame(slider_area, bg="#0f172a")
            row.pack(fill="x", pady=12)

            name_lbl = tk.Label(row, text=label, font=("Segoe UI", 12),
                               fg="#e2e8f0", bg="#0f172a", width=8)
            name_lbl.pack(side="left")

            val_entry = tk.Entry(
                row, font=("Segoe UI", 12, "bold"), width=6,
                bg="#1e293b", fg=color, insertbackground=color,
                relief="flat", borderwidth=0, highlightthickness=0,
                justify="center",
            )
            val_entry.pack(side="left", padx=(0, 16))
            val_entry.insert(0, f"{PRESETS['平衡'][i]:.2f}")
            val_entry.bind("<FocusOut>", lambda e, idx=i: self._on_entry_change(idx))
            val_entry.bind("<Return>", lambda e, idx=i: self._on_entry_change(idx))
            self._val_entries.append(val_entry)

            scale = WeightSlider(row, color=color,
                                command=lambda v, idx=i: self._on_slider_change(idx, v))
            scale.set(int(PRESETS['平衡'][i] * 100))
            scale.pack(side="left", padx=(0, 12))
            self._sliders.append(scale)

            pct_lbl = tk.Label(row, text="35%", font=("Segoe UI", 12, "bold"),
                               fg=color, bg="#0f172a", width=6, anchor="w")
            pct_lbl.pack(side="left")
            self._pct_labels.append(pct_lbl)

        bar_frame = tk.Frame(self.root, bg="#0f172a")
        bar_frame.pack(fill="x", padx=40, pady=(24, 0))

        self._bar_canvas = tk.Canvas(bar_frame, width=560, height=24, 
                                     bg="#0f172a", highlightthickness=0)
        self._bar_canvas.pack(fill="x")

        adv_frame = tk.Frame(self.root, bg="#0f172a")
        adv_frame.pack(fill="x", padx=40, pady=(20, 0))

        tk.Label(adv_frame, text="时间上限", font=("Segoe UI", 10),
                 fg="#64748b", bg="#0f172a").pack(side="left", padx=(0, 8))
        self._max_runtime_var = tk.StringVar(value="60")
        e1 = tk.Entry(adv_frame, textvariable=self._max_runtime_var,
                      font=("Segoe UI", 11), width=6,
                      bg="#1e293b", fg="#e2e8f0", insertbackground="#3b82f6",
                      relief="flat", justify="center")
        e1.pack(side="left", padx=(0, 8))
        tk.Label(adv_frame, text="秒", font=("Segoe UI", 10),
                 fg="#64748b", bg="#0f172a").pack(side="left", padx=(0, 32))

        tk.Label(adv_frame, text="内存上限", font=("Segoe UI", 10),
                 fg="#64748b", bg="#0f172a").pack(side="left", padx=(0, 8))
        self._max_memory_var = tk.StringVar(value="2048")
        e2 = tk.Entry(adv_frame, textvariable=self._max_memory_var,
                      font=("Segoe UI", 11), width=6,
                      bg="#1e293b", fg="#e2e8f0", insertbackground="#3b82f6",
                      relief="flat", justify="center")
        e2.pack(side="left", padx=(0, 8))
        tk.Label(adv_frame, text="MB", font=("Segoe UI", 10),
                 fg="#64748b", bg="#0f172a").pack(side="left")

        btn_frame = tk.Frame(self.root, bg="#0f172a")
        btn_frame.pack(fill="x", padx=40, pady=(32, 32))

        skip_btn = FlatButton(
            btn_frame, text="使用默认",
            bg="#1e293b", hover_bg="#334155",
            fg="#94a3b8", hover_fg="white",
            font=("Segoe UI", 11),
            width=120, height=40,
            command=self._use_default,
        )
        skip_btn.pack(side="left")

        start_btn = FlatButton(
            btn_frame, text="开始评测",
            bg="#3b82f6", hover_bg="#60a5fa",
            fg="white", hover_fg="white",
            font=("Segoe UI", 12, "bold"),
            width=140, height=44,
            command=self._start,
        )
        start_btn.pack(side="right")

        self._apply_preset("平衡")

    def _apply_preset(self, name: str):
        values = PRESETS[name]
        for slider, entry, val in zip(self._sliders, self._val_entries, values):
            slider.set(int(val * 100))
            entry.delete(0, tk.END)
            entry.insert(0, f"{val:.2f}")
        
        for n, btn in self._preset_buttons.items():
            if n == name:
                btn._bg = "#3b82f6"
                btn._hover_bg = "#60a5fa"
                btn._fg = "white"
                btn._hover_fg = "white"
            else:
                btn._bg = "#1e293b"
                btn._hover_bg = "#334155"
                btn._fg = "#cbd5e1"
                btn._hover_fg = "white"
            btn.draw()
        
        self._update_display()

    def _on_slider_change(self, idx: int, value: int):
        for btn in self._preset_buttons.values():
            btn._bg = "#1e293b"
            btn._hover_bg = "#334155"
            btn._fg = "#cbd5e1"
            btn._hover_fg = "white"
            btn.draw()
        
        self._val_entries[idx].delete(0, tk.END)
        self._val_entries[idx].insert(0, f"{value / 100.0:.2f}")
        self._update_display()

    def _on_entry_change(self, idx: int):
        for btn in self._preset_buttons.values():
            btn._bg = "#1e293b"
            btn._hover_bg = "#334155"
            btn._fg = "#cbd5e1"
            btn._hover_fg = "white"
            btn.draw()
        
        try:
            val = float(self._val_entries[idx].get())
            val = max(0.0, min(1.0, val))
            self._sliders[idx].set(int(val * 100))
            self._val_entries[idx].delete(0, tk.END)
            self._val_entries[idx].insert(0, f"{val:.2f}")
        except ValueError:
            raw = self._sliders[idx].get() / 100.0
            self._val_entries[idx].delete(0, tk.END)
            self._val_entries[idx].insert(0, f"{raw:.2f}")
        
        self._update_display()

    def _update_display(self):
        raw = [s.get() / 100.0 for s in self._sliders]
        total = sum(raw)
        normalized = [v / max(total, 1e-10) for v in raw]

        for i in range(4):
            self._pct_labels[i].configure(text=f"{normalized[i]:.0%}")

        self._bar_canvas.delete("all")
        bw, bh = 560, 24
        x = 0
        for i in range(4):
            w = max(int(normalized[i] * bw), 2) if normalized[i] > 0.005 else 0
            if w > 0:
                self._bar_canvas.create_rectangle(x, 0, x + w, bh, 
                                                  fill=DIM_COLORS[i], outline="")
                x += w

    def _get_weight_config(self) -> WeightConfig:
        raw = [s.get() / 100.0 for s in self._sliders]
        try:
            max_runtime = float(self._max_runtime_var.get())
        except ValueError:
            max_runtime = 60.0
        try:
            max_memory = float(self._max_memory_var.get())
        except ValueError:
            max_memory = 2048.0
        return WeightConfig(
            w_accuracy=raw[0], w_speed=raw[1],
            w_memory=raw[2], w_success=raw[3],
            max_runtime=max_runtime, max_memory=max_memory,
        )

    def _start(self):
        self.result = self._get_weight_config()
        self.root.destroy()

    def _use_default(self):
        self.result = WeightConfig()
        self.root.destroy()

    def run(self) -> WeightConfig:
        self.root.mainloop()
        return self.result if self.result is not None else WeightConfig()


def launch_weight_ui() -> WeightConfig:
    window = WeightConfigWindow()
    return window.run()


if __name__ == "__main__":
    cfg = launch_weight_ui()
    print(f"权重: 精度={cfg.w_accuracy:.0%} 速度={cfg.w_speed:.0%} 内存={cfg.w_memory:.0%} 成功率={cfg.w_success:.0%}")
    print(f"时间上限={cfg.max_runtime}s 内存上限={cfg.max_memory}MB")
