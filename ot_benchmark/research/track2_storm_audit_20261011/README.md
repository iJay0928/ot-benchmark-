# 方向2 STORM最小审计

阶段为P1最小可运行结果，非完整图像基线复现；不报告AIGC得分或算法创新。研究合同见`model_contract.md`，冻结配置见`protocol_minimal.json`。原始源码不随仓库分发，使用哈希锁定的官方版本；源码变化时拒绝运行。

## 运行

需要Python、PyTorch、pytest；本地使用Python3.12.10、PyTorch2.11.0+cu128。参考求解器只支持正质量输入，尚不处理圆形目标的零质量求解梯度。

```powershell
git -c core.autocrlf=true clone https://github.com/MICV-yonsei/STORM.git upstream/STORM
git -C upstream/STORM checkout 88dbc3aca9944593d9648534cdfc7bbd6cda6224
python -m pytest test_transport_reference.py -q -p no:cacheprovider
python audit_storm.py --upstream upstream/STORM --output results_reproduction --device cpu
```

命令从本目录执行。本轮上游字节哈希锁定Windows的CRLF工作树，克隆命令显式设置换行策略；Git原始LF blob SHA256为`1152ca60fac30176c5a56dc26bccfdc8045bcae24e77b5d68f7aa08b720c735a`，CRLF工作树为协议中的`9a7756d2...`。仅换行不同也会被当前加载器拒绝；不要通过编辑上游绕过哈希检查。自身源码固定LF，冻结结果保留原始字节。

原版圆形目标使用硬编码CUDA，只有该环境有CUDA时才执行相反方向的完整损失探针；无CUDA时明确记`not_run`。`--device`控制主运输审计，方向探针的设备单独记录为CUDA。结构输入不是真实图像注意力。

新输出路径必须不存在，不覆盖旧证据。输出含冻结输入、协议、摘要、CSV、环境和源码哈希。边缘残差达标仅证明数值可行性，不是原始对偶gap证书；本轮不作新方法性能比较。

## 依据与后续

- [STORM作者原文](https://arxiv.org/html/2503.22168v1)第4.2节及附录C.3。
- [官方源码](https://github.com/MICV-yonsei/STORM)的成本、Sinkhorn和方向路由。
- 下一步是兼容生成环境、真实注意力复现、论文一致成本和非OT解析强对照，然后才实施对象完整性候选与最终图像指标评测。

数学建模技能本轮仅启用编程P1检查。独立P1已通过，13项测试通过，证据见`P1_review.md`，研究含义见`审计结果与下一步.md`；完整P2的图像与图表交付尚未进行。

`requirements_image.txt`仅记录下一阶段的隔离环境依赖，不是已验证的完整图像基线。不要向当前研究venv直接降级安装；保留已可用的CUDA版Torch，缓存使用独立workspace目录。生成环境、模型与未完成步骤见`进度记录.md`。本次自动续跑创建被工具审批策略拒绝，未成功设置定时任务。
