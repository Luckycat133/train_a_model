# 训练规范与工程边界

本文件替代旧版“2025–2026 最佳实践”中未经本项目验证的性能数字、硬件处方和不匹配的 API 示例。基线审查日期：2026-11-15；项目仍为实验性、准备归档。

交付事实以 [ROADMAP](ROADMAP.md) 为准；缺陷证据见 [审查](docs/development/AUDIT_2026-11-15.md)；以下是**必须满足的规范，不是当前代码全部已满足的能力**。

## 1. 可复现输入先于训练

每次运行保存源码 commit、依赖版本、设备/精度、seed、解析后的配置、数据清单/切分与 tokenizer SHA-256。配置错误或资源不匹配应在构建模型前退出，不能静默切换到另一个 tokenizer、默认参数或随机初始化。

保留原始古文，记录清洗与繁简转换策略，保护诗行/段落边界。按作品、来源及近重复簇划分 train/validation/test，再切块和增强。Tokenizer 只在 train 上构建。未知字率、特殊 token、ID 连续性、编码/解码与变体处理均需测试。

## 2. 损失与数值语义

训练输入是整数 token ID。预训练与 SFT 的 next-token 位移只做一次；SFT prompt/pad 不计入损失，允许监督的 assistant token 和 EOS 才计入。空监督样本不得产生伪成功或 NaN 后被静默跳过。

验证 NLL 按有效 token 数归约；若报告 perplexity，应使用不含辅助损失和 label smoothing 的可解释 NLL。不同 tokenizer 的 token-level perplexity 不直接横比；同时报告字/字节归一指标和任务指标。

梯度累积按实际更新次数及有效 token 计算，覆盖尾部窗口。AMP 溢出跳过 optimizer update 时不得把 scheduler 当作已成功更新继续推进。零有效 batch/零参数更新是失败，不是 loss=0 的成功实验。

## 3. 注意力与可选优化

先通过 CPU FP32 参考对齐、因果性、过去依赖及 cache 等价测试。SDPA bool mask 的 True 表示允许参与注意力；additive mask 使用 0/-inf，不能相互混用。RoPE 解码位置需包含历史长度；padding、滑窗和非方形查询必须统一处理。

只有实际测得使用何种后端才能声称 Flash Attention 生效。CPU、CUDA FP32/BF16/FP16、MPS、checkpoint、compile 和多卡分别验收，不相互推定。MoE/MLA/YaRN 名称与配置项不构成交付证据。

## 4. AMP、保存和恢复

使用与支持版本匹配的 `torch.amp.autocast(device_type, ...)`。CPU FP32 或不使用 scaler 的路径执行普通 backward；FP16 的 scaler、BF16 的策略独立配置和验收，不能给 GradScaler 添加 `bf16=` 参数。

`torch.save(state, path)` 不接受 `weights_only`；加载时使用审定的格式和安全选项，不对未知来源 pickle 关闭安全限制来“兼容”。保存模型、optimizer、scheduler、适用的 scaler、global optimizer step、RNG、数据游标/采样状态及数据/tokenizer/config 身份。明确支持 epoch 边界还是 mid-epoch 恢复，并拒绝超出能力的恢复请求。

checkpoint 先写临时文件、验证后原子替换；中断不能覆盖有效权重。`best` 来自明确的验证指标改善，不是“最后一个 epoch”。查找规则与保存规则一致。连续训练和保存恢复后的等价性是验收项，不能只检查文件存在。

## 5. 配置、入口与依赖

现有 CLI 接受 argparse 对应的平铺 YAML；显式 CLI 覆盖 YAML。不要把嵌套 `model/training` 示例直接交给该入口。`src/config.py` 与 `src/config/` 并存是待消除的歧义；修改配置前验证实际导入路径。

Accelerate 是另一套原型 trainer 的可选依赖，不是当前 CLI 已完成的分布式能力。未来采用时按官方 API 使用 prepare/accumulate/backward 与 optimizer.step/zero_grad，并测量真实更新数和梯度归一；不重复除以 accumulation steps。

依赖下界不等于经过验证的兼容矩阵。新支持版本须在干净环境安装并执行门禁，硬件缺失标为未执行，不得输出性能承诺。

## 6. 最小交付规则

执行 [G0–G6](docs/development/VALIDATION.md)，保留命令、环境、失败和跳过证据。当前 CI 只有 G0 范围；不能把本规范当成已经接入 CI 的门禁。

不提交权重、完整语料或密钥；模型外发须记录数据责任、配置、指标、限制和 SHA-256。历史权重清理需另行授权，不能为整洁擅自改写历史。
