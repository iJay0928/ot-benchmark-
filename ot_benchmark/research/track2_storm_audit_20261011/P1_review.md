# 独立P1验收回执（2026-10-11）

范围：数值/源码最小纵向切片q1、q2、q3。未覆盖真实图片、完整生成管线、最终AIGC效果及P2正式图表。

状态：PASS。质检者未参与代码编写或修正，只读权威文件，在`results_p1_independent_20261011`写入隔离复验结果。

## 冻结输入

- 官方STORM提交：`88dbc3aca9944593d9648534cdfc7bbd6cda6224`，上游工作树干净。
- `pipeline.py` SHA256：`9a7756d28af7c076a381d9e9956b9d1a152e675e264a72c34fb86906c2d9eeb0`。
- `model_contract.md`：`666094e47ffe99a0a31c3365e171e64983c2644217f48d46ee20eef26088bf06`。
- `protocol_minimal.json`：`3de0d203935b37378d808ed443e83c55d36127d81f763f4a794bd09d2ee00783`。
- `upstream_loader.py`：`9d6e51f3f2272fd1590a80565d24d1595d42ad2011d15201831c9851a4f9ee00`。
- `transport_reference.py`：`a4ab8b93b13ef027577aef64efae75fcaa48a53e0329dc7ce30fd4b19ea73648`。
- `audit_storm.py`：`0e2964dd5bb64891a1508acf743f5cc02afed4f40c629954ed40046026ff1e42`。
- `test_transport_reference.py`：`018fe0f91073388ccac9bf3cb548624460a28532c080995f1b772b91c41ba7fa`。
- 验收时README：`a25199acafa022db633c914d37e24e50b3e382dffbae58f49708aab54932fdbe`。
- 实际输入哈希：`f3228db8878fa4da9141e3259ec736ee57658b8f1619139f954aceac28dd560a`。

## 独立运行证据

从本模块目录，用仓库`.venv/Scripts/python.exe`，设置`PYTHONDONTWRITEBYTECODE=1`：

```powershell
python -m pytest test_transport_reference.py -q -p no:cacheprovider
python audit_storm.py --upstream D:/桌面/ot_benchmark_work/track2_workspace_20261011/upstream/STORM --output results_p1_independent_20261011 --device cpu
```

两条命令退出码均为0；13项测试通过。独立输出的inputs、protocol、summary、transport_summary及source_manifest与作者输出逐字节哈希一致（耗时所在environment不要求相同）。CUDA原始损失和梯度探针实际执行，并非仅通过源码搜索作判断。

输入16×16，严格为正且质量约1；网格距离单位是格点距离平方，范围[0,450]；lambda D范围[0,4.5]。独立检查原sinkhorn代码文件与首行指向锁定上游pipeline.py:345。

q1：重复列差0，参考外积误差2.90e-18，解析线性值误差5.55e-16，参考残差1.17e-15。原版float32质量0.9960010052，行列L1残差0.1219952181，不能对不满足边缘条件的原版结果直接套用可行计划结论。

q2：固定注意力/token输入，left/right和above/below损失差与梯度差均0；梯度有限且非零。均匀图above成本行变化0、列变化0.390625，与源码332行转置相符。此结果不等于不同文本生成的最终图片相同。

q3：混合计划与lambda D计划最大差2.33e-17，边缘残差约6.41e-11，低于1e-10。质检者追加只读核验：线性值恒等误差6.96e-11；source/reference梯度恒等误差1.21e-10/1.00e-11，梯度有限且非零。

## 发现与适用范围

P0、P1：无阻塞项。P2建议：权威audit摘要目前只持久化row-only梯度对照；后续增强mixed梯度持久化证据。上述追加q3梯度数值是独立复核证据，不伪装成已有summary字段。

无需强制返工。代码、合同、输入或参数实质变更使本PASS失效，须以新输出目录复验。新增环境或图像实验不属于本回执，不可据此声称图像复现/P2已经通过。
