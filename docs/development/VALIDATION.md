# 验证门禁与证据规范

更新日期：2026-11-15。当前代码状态见 [ROADMAP](../../ROADMAP.md)，问题编号见 [审查](AUDIT_2026-11-15.md)。

**以下 G1–G6 是必须实现并执行的验收要求，不是本次已增加的测试或 CI job。当前 CI 仅覆盖 G0 中的源码编译和 CLI 测试。**

## 结果与证据格式

每个 gate 记录 `PASS / FAIL / BLOCKED / NOT_RUN / NOT_APPLICABLE`，附源码 commit、完整命令、Python/依赖、硬件/精度、seed、退出码、测试数量/跳过原因和日志/产物。PASS 不能只来自测试文件存在或 mock。NOT_APPLICABLE 必须说明能力不在本次发布范围，不能用于默认训练路径。

模型产物不进 Git；保存可共享的摘要、配置、校验和和脱敏日志。失败结果同样保留，不能删失败用例让 dashboard 变绿。过往 CI 只对其对应 commit 和命令有效。

## G0 — 入口、依赖与测试收集

现有轻量命令：

```bash
python -m pip install -r requirements-ci.txt
python -m compileall -q src
python -m pytest -q tests/test_cli_contract.py
python -m src.run --help
```

基线历史 CI 只执行前面安装/编译/pytest 三项；`--help` 在其中一个测试里运行。该文件有两个测试，另一个仅检查 quick preset 的参数，并未验证训练随机性。

后续实现需要在依赖合规的环境收集两个目录并保留结果：

```bash
python -m pytest --collect-only -q test tests
```

该收集命令在基线未由本次审查执行；架构测试引用缺失 MLA 类是已知静态阻断。默认路径测试必须可收集；可选 MLA 测试应单独隔离并说明未实现，不能连带跳过基础注意力测试。禁止把收集错误改写成“0 tests passed”。

配置验收包括 CLI > YAML > 默认值优先级、拒绝未知键和非法类型、解析配置回显、实际导入的 `src.config.__file__`、tokenizer 与模型词表一致性。不能静默丢弃嵌套配置。干净环境安装须满足仓库要求；仅能运行低版本探针不算通过。

## G1 — 数据与 Tokenizer

必须用真实 tokenizer/dataset，覆盖已知字、未知字、扩展汉字、异体/变体选择符、繁简、标点、空文本、诗行换行、混合文本、特殊 token 和异常 JSONL。

验收断言：

- `encode` 输出整数列表，tensor dtype 为 long，`0 <= id < vocab_size`；PAD/UNK/BOS/EOS 语义和 ID 固定，无重复 ID/稀疏 ID 越界。构建、保存、重载后的编码一致，记录词表及预处理策略指纹。
- 已覆盖字符不得因词组最大匹配而无故变成 UNK；明确无损 round-trip 或经批准的规范化规则。缺文件/格式错误不能静默换成取模字符 ID。
- 分别测试 tokenizer 存在/不存在/损坏，拒绝空数据、零有效 batch、全部过滤和 ID 越界；短语料重复只能作为显式标记的 smoke fixture 行为。
- 文档拼接、EOS、packing 跨文档 attention/loss 策略需明确。先做作品/来源/近重复簇隔离，再切块和增强；抽样检查分区交叉重复，Tokenizer 不读取 validation/test。
- 保存数据来源、适用授权/使用条件、原始与清洗后哈希、拒绝数量和原因、清洗版本/参数、split seed 与清单。抽查清洗 diff，不把公开可下载等同于可再分发。

默认清洗不能未经决定折叠诗行或损坏古文用字。繁简转换应有实际生效的反例测试，而非只检查输出类型。

## G2 — 模型数值与推理状态

使用小模型、固定 seed、CPU FP32、`.eval()`、dropout=0，选择 `sequence_length != num_heads`。与独立手写参考 attention 比较前向及梯度，初始要求 `atol=1e-5, rtol=1e-5`；调整容差需记录误差、原因与 dtype，不得仅为通过而放宽。

必须覆盖 MHA/GQA、legacy/modern、mask/no-mask、padding、滑窗、全遮蔽行和非法 head 配置。全遮蔽行必须有明确有限输出/拒绝策略，不能让 NaN 静默进入训练。

两种关键性质缺一不可：改变未来 token 不改变此前 logits；受控修改过去 token 必须影响允许依赖它的后续输出。错误地完全不做跨 token 注意力也能通过前一种测试，所以仅测“因果不泄漏”不足。

SDPA 采用 `[B,H,L,D]` 等明确契约；bool mask True=允许，additive mask 0/-inf；手写 fallback 必须数值等价，不能将 bool mask 直接加到 scores。

cache 测试比较完整前缀 forward 与 prefill+逐 token decode 的 logits，并覆盖 chunk decode、非零 past offset、padding 与窗口边界。RoPE 使用正确绝对位置，非方形 causal mask 对齐 query 的历史偏移；GQA 按未扩头 KV 保存。记录实际 cache 字节而不是由配置估算。

MLA 另需真实类、模型路由、压缩 cache 格式与位置编码设计，以及相同语义的 cached/full-prefix 等价性；与 MHA/GQA 比较质量和性能属于 G6，不能从压缩形状直接推导质量。

## G3 — 真实 CPU 训练、保存与恢复

关闭 AMP、compile、MoE、未验收 checkpointing，使用有版本的小数据及真实模型/optimizer，完成至少两次 optimizer update。loss、梯度和参数必须有限，至少一组非冻结参数变化。记录有效 token 数、更新数、学习率序列；零有效 batch 必须报错。

修复后执行当前真实 smoke 入口：

```bash
python -m src.run --quick --no_cuda
```

基线状态为 BLOCKED；该命令存在并不代表已通过。还需独立固定 seed 小样本过拟合测试，预算/成功阈值在运行前写入测试；不能用它替代 G2，也不能称为泛化质量。

checkpoint 需要真实写入、重载并比较 logits。连续 N 次更新，对比分段 K 次更新、保存/重载、再 N-K 次更新：比较参数、optimizer、scheduler、适用 scaler、global step、RNG 及采样游标。只恢复权重应标为 warm start，不是 resume。GPU/非确定性误差单独定义，CPU 不能靠跳过状态比较通过。

明确 checkpoint 支持的恢复边界；未支持 mid-epoch 时拒绝或清楚声明回退，而不是假装精确恢复。测试尾部梯度累积、中断保存、损坏文件、配置/tokenizer 指纹不匹配、best 指标和 latest 发现。best 必须由验证改善决定，文件写入须原子化，不覆盖最后有效状态。

## G4 — SFT 专项

前置 G0–G3。新 trainer 必须先通过真实构造函数与 forward 契约测试，覆盖 batch=1/2/3，避免把 `[B,L,V]` 沿 batch 维解包。

使用 instruction/output、prompt/completion、多轮 messages 的手工可核验样例。明确每条样本监督哪些 assistant turn；system/user/历史轮次的处理有固定规则，不能把目标答案再复制进同条样本的 prompt。

核对 logits/labels 只做一次 next-token shift；mask 排除 prompt/pad，EOS 和 response 边界正确。不同长度 padding collator、attention mask、截断、空 response、全 ignore labels 均有明确行为；tokenizer 参数与真实 API 对齐。

必须从已确认来源的预训练 checkpoint 初始化，验证加载身份，再完成 loss/backward/update/save/reload。损失变化只能说明训练闭环，任务收益还要通过 G6。

## G5 — 硬件、精度与优化

CPU FP32、CUDA FP32/BF16/FP16、MPS、gradient checkpointing、compile、Accelerate 多进程分别列结果。未提供硬件即 NOT_RUN，不能把 skip 汇总成跨设备通过。

优化前后比较前向、梯度、真实更新数、学习率与 checkpoint 可读性；AMP 溢出跳步不错误推进 scheduler；检查 checkpointing 的输入/返回类型及 RNG/dropout 语义。无支持证据时维持 opt-in，不默认启用。

Accelerate/DDP 使用真实多进程短运行验证设备放置、accumulate/sync、梯度缩放、有效 token 归约、尾部窗口、rank-safe checkpoint 与恢复。按累计窗口有效 token 总数归一，不能平均各 microbatch 的均值；框架已经做的缩放不得重复。

测量后端、显存峰值、吞吐与编译冷启动开销；SDPA 调用不等于 Flash kernel 成功使用。仅存在 Accelerate import 或 benchmark 脚本不算通过。

## G6 — 模型质量与性能声明

实验前冻结问题、基线、数据/tokenizer/配置哈希、seed、token/算力预算、主指标、护栏和停止条件。文言任务可从断句、释义/翻译、受控续写中选择一个首要目标；本规范不预先声称任何收益。

验证集用于选择，测试集留作最终评价；按有效 token 的 NLL 与独立任务指标分别报告。更换 tokenizer 时增加字/字节归一指标，不直接比较不兼容的 token PPL。训练 loss、过拟合样例、shape 测试均不构成质量证明。

性能对照必须同硬件/版本/精度/长度/batch/缓存策略，含 warmup、同步、重复测量、统计波动和正确性前置结果。不能把通用框架宣传的加速比作为本仓库成绩。发布时包含配置、限制、数据责任和外部权重校验和，权重不进入 Git。

## CI 落地与声明规则

下一代码 PR 应新增真实 CPU 合约/数值/smoke job；job 名称与测试文件在实现时确定，本文件不虚构已存在的 workflow。默认训练声明至少需 G0–G3；SFT 还需 G4；优化/设备声明需相应 G5；质量/加速声明需 G6。

发布或状态升级必须附结果清单。纯文档 PR 只能报告文档校验与历史/静态证据，不能关闭运行时阻断。

## API 核对来源

2026-11-15 核对官方文档：[SDPA](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)、[torch.save](https://docs.pytorch.org/docs/2.15/generated/torch.save.html)、[AMP](https://docs.pytorch.org/docs/2.15/amp.html)、[Accelerate accumulation](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)。这些是契约依据，不是本项目已经在文档版本上运行的证据；实际支持版本仍需记录并执行门禁。
