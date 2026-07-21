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

继续最新检查点：

```bash
python -m src.run --train your-data.jsonl --resume
```

查看全部真实可用参数：

```bash
python -m src.run --help
```

## 数据与权重边界

- `.pt`、`.pth`、`.ckpt` 和 `model_weights/` 中的新二进制文件不进入 Git。
- 当前没有经过独立验证的公开权重下载地址。
- 外部权重发布必须记录数据版本、配置、指标、限制与 SHA-256。
- 历史中的旧权重尚未清除；历史改写需要先完成备份与设备协调，当前 PR 不执行强推。

详见 [model_weights/README.md](model_weights/README.md)。

## 验证范围

当前 CI 运行源文件编译和轻量 CLI 合约测试。完整 PyTorch 训练仍需在合适的本地环境执行。现阶段不声称“最佳实践”、性能提升或模型质量，除非对应结果出现在可复现评估中。

最终结论、已知缺口和归档条件见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
