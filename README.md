# 灵猫墨韵 | Lingmao Moyun

> **Status: Experimental — Preparing for Archive**

古典中文语言模型训练工程实验。当前目标是保存可复现的工程经验并完成安全收尾，不再提交模型权重，也不把烟雾测试描述成模型质量证明。

## 快速验证

安装完整训练依赖后运行：

```bash
python -m pip install -r requirements.txt
python -m src.run --quick --no_cuda
```

`--quick` 使用仓库内公开样例，运行 1 个 epoch 的小模型并把输出写入已忽略的 `quick_runs/`。它只验证数据加载、模型构建、训练循环和保存路径，不代表实际生成质量。

自定义数据使用 JSONL，每行包含 `text`、`content`、`body` 或 `paragraphs` 字段之一：

```bash
python -m src.run --train your-data.jsonl --epochs 10 --save_dir quick_runs/custom
```

继续最新检查点（或使用 `--resume_from <checkpoint>` 精确指定某个检查点）：

```bash
python -m src.run --train your-data.jsonl --resume
```

查看全部真实可用参数（由 [src/run.py](src/run.py) 定义）：

```bash
python -m src.run --help
```

## 数据与权重边界

- `.pt`、`.pth`、`.ckpt` 和 `model_weights/` 中的新二进制文件不进入 Git。
- 只分发明确标注的微型基准 checkpoint（见 [制品范围与复现](model_weights/README.md)）；没有发布生产模型权重。
- 外部权重发布必须记录数据版本、配置、指标、限制与 SHA-256。
- 历史中的旧权重尚未清除；历史改写需要先完成备份与设备协调，当前 PR 不执行强推。

详见 [model_weights/README.md](model_weights/README.md)。

## 验证范围与入口

当前 [CI](.github/workflows/ci.yml) 含 CLI、Tokenizer ID、真实 CPU 训练及完成 epoch 恢复、MLA 数值和 SFT 门禁。历史 [验证回执](docs/development/VALIDATION.md) 仅适用于各自提交。以下命令使用仓库内小样本，不下载语料或调用 API：

```bash
python -m src.run --quick --no_cuda --use_mla --mla_latent_dim 8 --mla_num_latent_heads 2 --save_dir quick_runs/mla
python -m pytest -q tests/test_cpu_training.py tests/test_mla_cpu.py tests/test_sft_cpu_smoke.py
```

MLA 是现代模式的可选开关，默认架构及已有非 MLA 断点不变；不与滑窗同用。组数须整除查询头数与联合潜维。压缩 cache 是每层 `[B,T,G,R/G]` 联合潜变量与 `[B,T,P]` 旋转位置键，只有当 `R + P` 小于常规 K/V 宽度时才减少缓存元素。SFT 以 `SFTTrainer.from_pretrained` 的有界 CPU 测试为证，不把旧实验脚本、GPU/分布式或模型质量作为已验证能力。

- CLI 入口：[src/run.py](src/run.py)
- 配置常量与 Schema：[src/config.py](src/config.py) · [src/config/\_\_init\_\_.py](src/config/__init__.py)（注意仓库同时存在 `src/config.py` 与 `src/config/` 包，导入 `src.config` 时优先解析包入口）
- CLI 合约测试：[tests/test_cli_contract.py](tests/test_cli_contract.py)
- 最终结论、已知缺口和归档条件：[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)
- 版本变更记录：[CHANGELOG.md](CHANGELOG.md)
- 文档总索引与数据/实验指南：[docs/README.md](docs/README.md) · [docs/data/README.md](docs/data/README.md) · [docs/experiments/README.md](docs/experiments/README.md) · [docs/experiments/quickstart.md](docs/experiments/quickstart.md)

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
