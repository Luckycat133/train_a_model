# Lingmao Moyun — Final Experiment Report

**Status: Experimental — Archived**
**Last updated: 2026-07-19**

---

## 做了什么

### 项目目标

灵猫墨韵是一个探索古典中文方向 Transformer 语言模型训练的工程实验。项目实现了数据加载、分词器集成、训练循环、评估、检查点保存以及现代 Transformer 组件（RoPE、SwiGLU、GQA、MoE、Flash Attention、权重绑定）。

### 训练完成的内容

- **usable_model (5 epochs)**：6 层 modern Transformer，d_model=768，nhead=12，vocab_size=30000。使用本地古典中文语料训练 5 个 epoch。
- **usable_model_20ep (20 epochs)**：同架构，训练 20 个 epoch。

### 权重现状

模型权重文件（`.pt`）仅存在于本地磁盘 `model_weights/` 目录中：
- `usable_model/`：best_model.pt (14 MB)，final_model.pt (14 MB)，5 个检查点（各 42 MB）
- `usable_model_20ep/`：best_model.pt (14 MB)，final_model.pt (14 MB)，20 个检查点（各 42 MB）

Git 历史已通过 `git filter-repo` 清理，`.git` 对象存储从 965 MB 缩减至约 500 KB。所有 `.pt`、`.pth`、`.ckpt` 文件已从 Git 历史中移除。模型权重文件通过 Git LFS 追踪，LFS 对象缓存在 `.git/lfs/` 中。

### 外部存储

未发布到外部存储（Hugging Face Hub、GitHub Releases、对象存储）。Git LFS 已配置，可在推送时自动上传到 GitHub LFS 存储。

---

## 学到了什么

### 工程经验

1. **Git 不适合存储模型权重**：26 个检查点文件（每个 14-42 MB）使仓库膨胀到 965 MB。权重应使用专用存储。
2. **烟雾测试不等于质量评估**：`--quick` 模式仅验证数据加载、前向传播、反向传播和保存路径，不代表模型生成质量。
3. **CLI 示例必须可自动化执行**：`python -m src.run --help` 不应依赖重型训练库。修改后 `--help` 只需标准库即可运行。
4. **从零预训练需要充分准备**：缺少许可数据集、固定计算预算和评估套件的情况下，从零预训练大模型是不合理的。
5. **实验记录必须完整**：随机种子、数据版本、配置、验证指标、限制说明都必须记录。

### 技术探索

- 实现了 RoPE、SwiGLU、GQA、MoE、Flash Attention、权重绑定等现代 Transformer 组件。
- 这些组件在代码中正常工作，但缺少端到端的基准验证。

---

## 模型达到什么效果

| 评估项 | 状态 |
|---|---|
| 固定随机种子 | 未记录 |
| 数据集版本 | 未记录（仅公开 smoke fixture 有版本） |
| Loss 曲线 | 未独立验证 |
| 验证集指标 | 不可用 |
| 生成质量评估 | 未独立验证 |
| 权重 checksum / 外部 URL | 未发布 |
| CI | 通过（243 passed, 16 skipped） |

**不声称任何性能提升或模型质量**，除非对应结果出现在可复现评估中。

---

## 为什么不继续从零训练大模型

1. **计算资源不足**：从零预训练需要大量 GPU 小时。
2. **数据集许可和规模**：古典中文语料的收集、清洗、许可和版本管理需要大量工程工作。
3. **评估缺失**：没有建立固定的评估套件。
4. **成本效益**：微调现有开源模型比从零预训练更经济、更快速。

---

## 后续建议：微调还是继续预训练

### 推荐：微调（Fine-tuning）

- **适用场景**：有特定任务需求（如古典诗词生成、文言文翻译、古籍问答），且已有高质量标注数据。
- **优势**：成本低、见效快、可利用现有开源模型。
- **推荐基座**：Qwen3、DeepSeek-V3、LLaMA-3（均支持中文）。
- **方法**：LoRA/QLoRA 微调，或全参数 SFT。

### 不推荐：继续预训练

除非满足全部条件：独特大规模数据、充足计算预算、固定评估套件、可复现实验流程。

---

## 归档条件检查

| 条件 | 状态 |
|---|---|
| 权重不在 Git 历史 | 已完成 |
| quick 模式可运行 | 已完成 |
| 有最终实验报告 | 本文档 |
| 模型文件有外部存储地址 | Git LFS 已配置 |
| README 不夸大能力 | 已完成 |

---

## 归档决定

**仓库状态：Cleaned Experimental Repository → Archived**