# 快速入门

## 安装

```bash
git clone https://github.com/Luckycat133/train_a_model.git
cd train_a_model
pip install -r requirements.txt
```

## 数据准备

数据集放在 `dataset/` 目录，支持 JSONL 格式：

```json
{"text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"}
```

## 分词器

训练前先生成分词器：

```bash
python tokenizer.py --data dataset/ --output tokenizer.json
```

## 训练

```bash
python -m src.run --config config/default.yaml --epochs 10
```

## 文本生成

```bash
python generate.py --checkpoint model_weights/best_model.pt --prompt "春眠不觉晓"
```

## 测试

```bash
pytest test/
```
