# Scripts Directory

This directory contains executable scripts for the Lingmao Moyun project.

## Available Scripts

| Script | Description |
|--------|-------------|
| [data_pipeline.sh](data_pipeline.sh) | One-click data preparation pipeline |
| [quantize_model.py](quantize_model.py) | Model quantization utility |

## Data Pipeline

The `data_pipeline.sh` script provides a complete data preparation workflow:

```bash
# Run full pipeline (download → clean → merge → validate)
./data_pipeline.sh --full

# Download data
./data_pipeline.sh --download --source poetry

# Clean data
./data_pipeline.sh --clean -i raw.jsonl -o cleaned.jsonl

# Validate data
./data_pipeline.sh --validate -i dataset/merged.jsonl

# Show statistics
./data_pipeline.sh --stats
```

For detailed documentation, see [docs/data/README.md](../docs/data/README.md).

## Model Quantization

The `quantize_model.py` script supports model weight quantization for efficient inference:

```bash
python scripts/quantize_model.py --input model.pt --output quantized_model.pt
```

See [README_INFERENCE.md](../README_INFERENCE.md) for more details.
