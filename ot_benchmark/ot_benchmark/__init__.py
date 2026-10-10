"""
OT Benchmark v2.0 — 面向高维稀疏数据的最优传输算法自动评测系统
===============================================================
大学生创新创业训练计划项目
华南理工大学 · 计算机科学与工程学院

团队成员: 黄儒家、许林烨、王紫来、陈俊龙
指导老师: 何克晶

v2.0 新特性:
  - GPU 加速后端 (PyTorch / CuPy)
  - 稀疏 CSR 格式 OT 算法
  - 新增 Gaussian OT, Factored OT, Sparse Sinkhorn
  - 并行评测引擎
  - 华为云 OBS / ModelArts 集成
  - YAML 配置文件支持
  - 结构化日志系统
  - 完整单元测试
"""

__version__ = "2.0.0"
__author__ = "SCUT CS Innovation Team"
