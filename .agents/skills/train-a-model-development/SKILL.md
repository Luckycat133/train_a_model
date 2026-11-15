---
name: train-a-model-development
description: 审查、开发和验收 Luckycat133/train_a_model 古典中文训练实验；用于数据清洗、Tokenizer、Transformer/MLA、PyTorch 预训练与 SFT、checkpoint、测试及交付状态更新。
---

# train_a_model 项目开发 skill

## 适用范围与事实来源

这是项目专用 skill，不是通用 PyTorch 教程。审查基线为 `01a5b3b9faf04ff0a70bfdbbd2ee97f76c5a2c4f`，审查日期为 2026-11-15。后续任务先读取当前 HEAD 和 diff，再决定哪些结论仍成立；不要把本次快照永久当作最新事实。

从仓库根目录依次阅读：

1. `README.md`、`ROADMAP.md`：产品边界、交付状态和下一阶段。
2. `docs/development/AUDIT_2026-11-15.md`：已知阻断及证据级别。
3. `docs/development/VALIDATION.md`、`BEST_PRACTICES.md`、`docs/en/architecture.md`：验收门禁和契约。
4. 本次修改实际触及的源码、调用者、配置及测试；不能只读取 changelog 或 benchmark 结论。

通用 `pytorch-patterns` skill 是上游参考，不能覆盖本文件；不要改写其锁定哈希来登记本地 skill。

## 当前边界：不是已验收的完整训练平台

- 默认定位仍为 **Experimental — Preparing for Archive**。先修复并保存最小可复现实验，再由维护者决定是否继续研究。
- `src/run.py -> src/trainer.py -> LMDataset/SimpleTransformer` 是当前公开 CLI 路径。`src/training/{base_trainer,pretrain,sft_trainer}.py` 是另一条原型路径，不能默认替代前者。
- `SimpleTransformer` 包含 modern/legacy 分支。现代组件、缓存、MoE 和 checkpoint 开关的存在不代表数值正确。
- 当前 `src/model.py` 未定义 `MultiHeadLatentAttention`，但调试脚本与架构测试引用它。MLA 必须记为未交付，不能根据配置常量补写“已实现”。
- `ClassicalTokenizer.train()` 是 no-op；`build_tokenizer.py` 构建字符词表，不是已验收的 BPE/Unigram 训练器。
- CI 仅编译 `src` 并运行两个 CLI 契约测试。CPU 真训练、SFT、恢复一致性、GPU、多卡与模型质量尚无本次验收证据。

## 工作流程

### 1. 建立可追溯基线

记录 commit、未提交修改、Python/PyTorch/Accelerate 版本和设备；确认运行环境满足依赖。读取 `src.config.__file__`，不要默认 `src/config.py` 生效：同名 package 当前也存在。先做收集与小样本检查，不自动下载全量数据或启动训练。

### 2. 先写失败回归，再修改最小路径

定位输入、输出、调用者和损失计算的所有权。优先关闭审查 A01–A07；不同时扩展 MLA、MoE、RL 和服务端功能。任何迁移须有独立说明：保留哪个入口、如何兼容旧配置/权重、哪些能力仍不可用。

不得用 broad exception + 静默 fallback、删断言、全局 skip、放宽容差或伪造 mock 返回值掩盖错误。测试替身只证明被测接口；训练验收必须使用真实模型与真实优化器。

### 3. 保持数据和 Tokenizer 契约

`tokenize` 返回字符串；训练只能消费明确的整数 `encode` 输出。校验词表 ID、特殊 token、未知字策略与 tokenizer 指纹，缺失/损坏资源应失败退出；`ord(c) % 30000` 不能作为真实训练的静默替代。

清洗策略须声明繁简/异体字、扩展汉字、变体选择符、标点和诗行换行如何处理。保留原文及变换记录。先按作品/来源及近重复簇隔离数据，再做切块、packing 或增强；Tokenizer 训练只能使用训练分区。不能把重复片段的随机划分称为 held-out。

### 4. 固定模型和训练契约

- 当前默认 forward 返回 `[B,L,V]` logits；cache/aux 返回形式是显式开关，不得按 batch 维度解包。
- SDPA 输入必须明确序列/头维，布尔允许掩码与 additive mask 分开处理。legacy 也必须通过因果性测试。
- 验证未来不泄漏，也验证确实依赖过去；形状正确和 loss 下降都不足以证明注意力正确。
- cache 必须使用正确绝对位置和非方形因果掩码；GQA 缓存不应误存扩头后的张量；MLA 必须单独定义压缩状态与解码等价性。
- 预训练与 SFT 均只进行一次 next-token 位移。SFT 只监督允许的 assistant token；答案不能重复进入该样本的上下文。
- 梯度累积按真实有效 token 和更新步计数；尾部不完整窗口、AMP 跳步与多卡归约都需要回归。
- `torch.save` 不传 `weights_only`；加载安全策略、完整状态恢复和原子写入另行验收。推理权重加载不等于训练续跑。

### 5. 按门禁验收而不是按功能名勾选

执行 `docs/development/VALIDATION.md` 的 G0–G6 中与改动相关的门禁。先保证 CPU FP32 的 G0–G3；SFT 声明需 G4；GPU/优化声明需 G5；模型质量/性能声明需 G6。

当前已有命令：

```bash
python -m compileall -q src
python -m pytest -q tests/test_cli_contract.py
python -m src.run --help
```

下列是真训练验收目标，**审查基线存在阻断，不是已通过示例**：

```bash
python -m src.run --quick --no_cuda
```

不得编造 `--prepare`、`--eval`、`--sft` 或 CLI 不支持的配置选项。新增 CLI 需先补 parser 契约和执行 smoke。测试目录同时有 `test/` 与 `tests/`；收集失败属于失败，不能当作“不适用”。

### 6. 更新交付事实

同步路线图、实验报告、入口说明和对应审查项。记录成功/失败/跳过/未执行，而非单个“测试通过”结论。新状态必须包含命令、commit、环境和可检查产物；只写新测试未运行不能升级为已验证。

## 默认下一步

执行 ROADMAP 的 R1：Tokenizer-ID 接口、CPU 真训练与 checkpoint、注意力数值契约。R2 的 SFT 与 R3 的 MLA/规模实验都不能越过其前置门禁。没有许可证、数据清单、算力预算与固定评估集，不启动无边界从零预训练。

## 完成定义

变更能被最小回归解释，相关门禁有明确结果，文档不夸大能力；未修复项仍显式开放。不提交 `.pt/.pth/.ckpt`，不改写 Git 历史、不强推、不自动归档。最终报告区分工程验收与语言模型质量。
