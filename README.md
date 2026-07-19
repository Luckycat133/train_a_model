# Lingmao Moyun

> **Status: Cleaned Experimental Repository — Archived**

古典中文语言模型训练工程实验。仓库已清理归档，权重不再存储在 Git 中。

## 快速验证

```bash
pip install -r requirements-ci.txt
python -m src.run --help
pytest -q
```

安装完整训练依赖后运行烟雾测试：

```bash
pip install -r requirements.txt
python -m src.run --quick --no_cuda
```

`--quick` 使用 `examples/quick_train.jsonl` 中的公开样例，运行 1 个 epoch 的小模型并把输出写入已忽略的 `quick_runs/`。它只验证数据加载、模型构建、训练循环和保存路径，不代表实际生成质量。

## 数据与权重边界

- `.pt`、`.pth`、`.ckpt` 和 `model_weights/` 中的二进制文件不进入 Git。
- Git 历史已清理，不再包含任何权重文件。
- 当前没有经过独立验证的公开权重下载地址。
- 详见 [model_weights/README.md](model_weights/README.md)。

## 验证范围

当前 CI 运行源码编译和轻量 CLI 契约测试。完整 PyTorch 训练需在本地 GPU 环境执行。

## 最终结论

完整实验报告见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。