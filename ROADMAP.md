# 灵猫墨韵：交付状态与路线图

审查日期：2026-11-15。源码基线：`01a5b3b9faf04ff0a70bfdbbd2ee97f76c5a2c4f`。
定位：**Experimental — Preparing for Archive**。本次是规范与证据更新，不是训练代码修复，也未启用新的 CI 任务。

## 状态口径

“存在”仅指源码/入口存在；“阻断”表示发现会阻止执行或破坏语义的具体问题；“未验收”表示缺少目标环境证据；“已验证”必须限定命令、环境和 commit。不得用实现比例推算交付比例。详细证据见 [审查记录](docs/development/AUDIT_2026-11-15.md)，验收定义见 [门禁](docs/development/VALIDATION.md)。

## 当前交付矩阵

| 领域 | 实现事实 | 验收状态与边界 |
|---|---|---|
| CLI 与轻量 CI | `src.run`、quick preset、两个 CLI 测试存在 | 基线历史 CI 成功；只证明源码编译/CLI，不证明训练 |
| 数据清洗 | `processors/`、`src/data/` 存在；抽查了 `TextCleaner` | 部分实现，语料来源/授权/去重和切分未审计验收；繁转简与换行策略有缺口 |
| Tokenizer | 最大匹配原型、字符词表 builder、vocab JSON 读写 | 阻断：字符串 token 与整数 ID 契约冲突；字符词表不覆盖默认词组；`train()` no-op |
| 主预训练入口 | `src.trainer.train_model`、`LMDataset`、`SimpleTransformer` | 阻断：AMP/scaler、监控与保存路径问题；quick 未获真训练验收 |
| modern/legacy 注意力 | RoPE、SwiGLU、RMSNorm、GQA/SWA 分支存在 | 阻断：SDPA 轴、mask、cache 位置/头数；legacy 无显式因果 mask |
| MLA | 配置常量、调试脚本与测试引用存在 | **未交付**：`src/model.py` 无 `MultiHeadLatentAttention` 定义；不是“待优化的已实现 MLA” |
| 新 pretrain/SFT trainer | `src/training/` 类与数据原型存在 | 阻断：父类参数/forward 契约不一致；SFT 位移、答案泄漏、padding 和 tokenizer 参数不兼容 |
| 保存与恢复 | 两套 checkpoint helper 存在 | 阻断：保存 API 参数、文件发现规则和完整状态恢复；不能宣称精确续训 |
| MoE / gradient checkpointing | 实验分支存在 | 存在维度/API 契约缺陷；不能作为默认可靠路径 |
| GPU / MPS / Accelerate / compile | 相关分支、示例与 benchmark 存在 | 未验收；无本次目标硬件、精度或多卡正确性/性能证明 |
| 质量评估与权重发布 | 评估代码和历史实验文档存在 | 无独立验证的 held-out 指标、质量结论或公开权重 URL；不发布性能提升数字 |

## R0 — 校正事实与规范（本次文档交付）

新增项目专用 skill、状态矩阵、审查记录和验证规范；纠正 architecture、训练规范、README 与实验报告的当前承诺。通用上游 skill 和 lock 保持不变。

完成的是文档资产，**不是** A01–A11 的代码修复。G1–G6 仍待实现/执行，CI 仍只有 G0 基线覆盖。

## R1 — 恢复可信的最小工程基线（下一开发批次）

目标不是扩大模型，而是一个真实、因果正确、可保存并恢复的 CPU FP32 小实验。

| 顺序 | 工作包 | 必须提交的验收证据 |
|---|---|---|
| R1a | 统一配置来源与 tokenizer 整数 ID 接口；拒绝静默 fallback；固定小数据及 seed | G0/G1；词表/ID 范围、未知字、round-trip、错误路径、有效样本数量 |
| R1b | 修复 SDPA 轴与 mask，legacy 因果性、RoPE/cache/GQA；基础模型默认不启用未验收优化 | G2；手写参考前向/梯度对齐、过去依赖、未来不泄漏、cache 等价 |
| R1c | 修复 CPU/AMP 分支、监控、checkpoint 写入/发现/恢复与更新步计数 | G3；真实至少两次更新，保存重载，连续训练对比分段恢复 |
| R1d | CI 加入真实 CPU 模型/数据/训练回归；完整收集两个测试目录并管理硬件分类 | 明确测试清单、失败/跳过原因及日志，不用缩小收集范围伪造全绿 |

R1a–c 需共同闭环后才能把 `--quick` 标记为通过。支持环境必须满足依赖并记录版本；本次独立 2.10 CPU 探针不是该验收结果。若选择归档路线，R1 加完整限制记录是最小工程收尾目标；历史清理仍需独立授权。

## R2 — 小型、可测量的 SFT（继续开发后优先）

前置：G0–G3 通过，维护者确认继续研究而不是只做归档收尾。

统一主 trainer/原型 trainer 的接口，先修复构造函数和 logits 契约；补可变长 collator、assistant-only 标签、一次位移、EOS、截断与空监督处理。完成真实预训练 checkpoint 到 SFT 的加载，记录初始化来源，禁止从随机初始化运行后称为微调。

完成标准：G4；至少两种长度、batch=1/2/3、多轮角色、故意泄漏的反例；真实 loss/backward/update/save/reload，prompt/answer 边界可人工核对。再以固定小评估集比较基线与 SFT，不用训练集 loss 代替收益。

## R3 — 架构研究与规模化（条件性计划，不是承诺）

MLA 若仍是研究目标：先提交设计决策，定义 latent cache、位置编码、checkpoint 格式和 MHA/GQA 对照；实现实际模型路由、prefill/decode 等价测试及真实 cache 字节测量。未通过 G2/G5 前不得称为压缩收益或推理加速。

BPE/Unigram 使用何种后端、词表大小与 byte fallback 要通过 G1/G6 的语料覆盖和等预算对照决定，不能把最大匹配原型改名当作实现。MoE、RL、量化、compile、多卡和服务化延后，分别设置独立门禁。

从零预训练扩大规模之前，必须先冻结合法数据版本、token 预算、硬件/费用上限、停机标准与 held-out 评估。性能优化以正确基线为参照，不照抄硬件建议或“1.5x”宣传值。

## 归档与重启决策

归档验收要求：权重明确不可用或提供经审核的外链/哈希；smoke 必须在合规环境实测通过，失败时记录事实并保持门禁阻断；最终限制与数据责任记录完整；历史权重清理须在另行授权、备份和协调后完成，未完成时保持该条件开放。不得未经授权强推或改变仓库归档状态。

重启研究前：维护者明确目标是教学复现、文言任务质量还是 MLA 架构实验，并选择一个作为首要目标。默认建议是先完成 R1，再做有上限的 R2，而不是并行堆叠新架构。
