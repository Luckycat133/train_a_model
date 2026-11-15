# Repository development instructions

本项目是灵猫墨韵（Lingmao Moyun），古典中文语言模型训练工程实验。

## 开始工作前

先读 [项目开发 skill](.agents/skills/train-a-model-development/SKILL.md)，再读 [路线图](ROADMAP.md)、[验证门禁](docs/development/VALIDATION.md) 与 [2026-11-15 审查](docs/development/AUDIT_2026-11-15.md)。涉及架构时读 [architecture](docs/en/architecture.md)，涉及训练时读 [训练规范](BEST_PRACTICES.md)。

项目专用规范优先于 `.agents/skills/pytorch-patterns/SKILL.md` 的通用示例。后者是 `skills-lock.json` 记录的上游依赖；不要为本项目改动伪造其上游哈希。项目 skill 由仓库直接维护，不冒充上游安装项。

## 不可越过的边界

- 保持 `Experimental — Preparing for Archive` 定位，除非维护者另行明确改变目标。可以修复可复现性；不要自动启动大规模预训练、下载全量语料或购买算力。
- 文档状态分开记录“源码存在”“测试已执行”“目标环境验收通过”。不能将 CI 绿色、测试名称、配置开关或测试替身当作功能交付。
- 真实 CLI 当前是 `python -m src.run`，调用 `src/trainer.py`；`src/training/` 是尚未接通、存在契约阻断的另一套实现。不要把它们混写成统一已完成平台。
- 不提交新权重、私有语料、密钥或缓存。历史清理、强推、归档仓库与外部权重发布均需单独授权；不能通过放宽 `.gitignore` 解决交付。
- 每次修改应包含对应回归测试与门禁证据。文档修订可只校验文档，但不得因此关闭代码阻断项。

## 交付报告

报告基线/产出 commit、修改范围、真实命令、环境版本、退出码、失败/跳过原因、产物位置与剩余阻断。未执行的命令明确标为未执行；只更新文档时明确代码与 CI 未修复。
