# 灵猫墨韵 | Lingmao Moyun

> **Status: Experimental — Preparing for Archive**

古典中文语言模型训练工程实验。当前目标是保存可复现的工程经验并完成安全收尾，不再提交模型权重，也不把烟雾测试描述成模型质量证明。

## 当前事实与开发入口

2026-11-15 对源码 `01a5b3b9faf04ff0a70bfdbbd2ee97f76c5a2c4f` 的审查发现 Tokenizer-ID、注意力数值、训练/保存和 SFT 契约阻断。MLA 的测试/调试引用存在，但当前模型文件没有对应实现。**不能将本仓库描述为已验收的完整预训练/SFT/MLA 平台。**

开发前阅读 [项目 skill](.agents/skills/train-a-model-development/SKILL.md)、[路线图](ROADMAP.md)、[验证门禁](docs/development/VALIDATION.md) 与 [审查记录](docs/development/AUDIT_2026-11-15.md)。训练规则见 [BEST_PRACTICES.md](BEST_PRACTICES.md)，实际调用路径见 [architecture](docs/en/architecture.md)。本次规范修订不修改训练代码或 CI。旧语言版 README、研究文档和历史 changelog 不能覆盖当前验收状态。

完整文档索引见 [`docs/README.md`](docs/README.md)，变更记录见 [`CHANGELOG.md`](CHANGELOG.md)。

## 轻量验证与真实训练目标

当前 CI 范围是：

```bash
python -m pip install -r requirements-ci.txt
python -m compileall -q src
python -m pytest -q tests/test_cli_contract.py
```

基线已有该范围的历史成功记录；两个 CLI 测试不执行真实训练。查看 parser 的实际参数：

```bash
python -m src.run --help
```

下列仍是修复后必须通过的真实 smoke 命令，**当前审查基线存在阻断，未确认可成功跑完**：

```bash
python -m pip install -r requirements.txt
python -m src.run --quick --no_cuda
```

`--quick` 配置为公开样例、1 个 epoch 的小模型，目标输出目录为已忽略的 `quick_runs/`。它通过后也只证明工程流程，不证明生成质量或精确恢复。

CLI 支持 JSONL 中的 `text`、`content`、`body` 或 `paragraphs` 字段，并提供 `--train`（`--train_file` 别名）、`--save_dir` 与 `--resume`。这些参数存在不代表对应训练/恢复流程已验收；使用前先关闭路线图中的阻断项。

## 数据与权重边界

- `.pt`、`.pth`、`.ckpt` 和 `model_weights/` 中的新二进制文件不进入 Git。
- 当前没有经过独立验证的公开权重下载地址。
- 外部权重发布必须记录数据版本、配置、指标、限制与 SHA-256。
- 历史中的旧权重尚未清除；历史改写需要先完成备份与设备协调，本次文档修订不执行强推。

详见 [model_weights/README.md](model_weights/README.md)。

## 验证范围

完整 PyTorch 流程须在满足依赖的目标环境执行。审查中的静态发现、独立 API 探针、历史轻量 CI 和真训练验收分开记录。现阶段不声称性能提升或模型质量，除非对应结果出现在可复现评估中。

最终限制和归档条件见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
