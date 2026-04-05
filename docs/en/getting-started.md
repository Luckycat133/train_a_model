# Getting Started

## Installation

```bash
git clone https://github.com/Luckycat133/train_a_model.git
cd train_a_model
pip install -r requirements.txt
```

## Data Preparation

Dataset expects JSONL files in `dataset/`. Each line should be a JSON object with a `text` field:

```json
{"text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"}
```

## Tokenizer

Before training, prepare the tokenizer:

```bash
python tokenizer.py --data dataset/ --output tokenizer.json
```

## Training

```bash
python -m src.run --config config/default.yaml --epochs 10
```

## Generating

```bash
python generate.py --checkpoint model_weights/best_model.pt --prompt "春眠不觉晓"
```

## Testing

```bash
pytest test/
```
