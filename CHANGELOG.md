# 更新日志

本项目遵循[语义化版本](https://semver.org/lang/zh-CN/)规范。

## [未发布]

### 新增

- 轻量 CLI 合约测试 `tests/test_cli_contract.py` 与 GitHub Actions 基线 CI（`.github/workflows/ci.yml`：源码编译 `python -m compileall -q src` + `pytest -q tests/test_cli_contract.py`）。
- `src/run.py` 快捷入口参数与稳定别名：`--quick`（基于 `examples/quick_train.jsonl` 的 1 epoch CPU 烟雾训练）、`--train`（`--train_file` 别名）、`--save_dir`（`--model_save_dir` 别名）、`--resume`、`--resume_from`、`--clean_before_run` 与 `--config`。
- 归档准备说明与证据边界文档：`EXPERIMENT_REPORT.md`、`model_weights/README.md`。
- 项目级 Agent 入口与技能路由：`AGENTS.md`、`CLAUDE.md`、`.agents/SKILLS.md` 与 `lingmao-experiment-closeout` 等项目级 skills。

### 变更

- 将 `README.md` 收敛为归档前可复现工程入口，明确烟雾测试与真实质量评估、以及权重不入库的边界。
- 修复 `docs/data/README.md`、`docs/experiments/README.md` 与 `docs/experiments/quickstart.md` 中的相对路径与失效 `file:///workspace/` 链接。

## [0.8.5] - 2026-04-05

### 新增

- ✨ **Modern模式架构**：RoPE + SwiGLU + GQA支持
- 🔧 **实验平台**：完整的配置管理和验证系统
- 📊 **评估模块**：多样性和LM指标评估
- 🎯 **量化工具**：AWQ量化支持
- 📝 **多语言文档**：中英文双语文档

### 修复

- 修复了ClassicalTokenizer的贪婪最大匹配问题
- 优化了数据加载的内存使用
- 修复了梯度检查点的实现

### 优化

- 实现了内存映射数据集
- 优化了训练循环性能
- 改进了日志记录

## [0.1.0] - 2025-04-23

### 新增

- 初始版本的数据处理脚本
- 基础的配置文件系统
- 日志记录功能
- 命令行参数解析
