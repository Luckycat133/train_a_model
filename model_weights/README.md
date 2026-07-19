# Model Weights

Model binaries are **not** stored in Git. Git history has been cleaned of all `.pt`, `.pth`, and `.ckpt` files. Weight files are tracked via Git LFS.

## Current Inventory

Weight files exist on local disk in `model_weights/`. No verified external download URL is available.

### 1. usable_model (5 epochs)

| Field | Value |
|---|---|
| Model | `SimpleTransformer` (modern mode), 6 layers |
| Architecture | RoPE, GQA, SwiGLU FFN, RMSNorm, Flash Attention |
| d_model | 768 |
| nhead | 12 |
| num_layers | 6 |
| dim_feedforward | 3072 |
| vocab_size | 30000 |
| Mode | modern |
| Epochs | 5 |
| Training dataset | 古典中文语料 (local JSONL, version not recorded) |
| Random seed | Not recorded |
| Validation metrics | Not available |
| Best model file | `best_model.pt` (14 MB) |
| Best model SHA-256 | `53ea1ae14250fb30e4be28dc63eb7b88020aedbc4b356415740c893708724ae7` |
| Final model file | `final_model.pt` (14 MB) |
| Final model SHA-256 | `a3236b22829a21db94bcb507554c06435bb1a385ed2f522f8a4891c3f1e9ea8c` |

### 2. usable_model_20ep (20 epochs)

| Field | Value |
|---|---|
| Model | Same architecture as above |
| Epochs | 20 |
| Best model file | `best_model.pt` (14 MB) |
| Best model SHA-256 | `8a510c6264fde3c19eb71f667770ae007940b763fb507a748a8a088d0f441b51` |
| Final model file | `final_model.pt` (14 MB) |
| Final model SHA-256 | `552479bc7b611968aaf321f1c2c67070c8b978d989a913310f368752c22c85a0` |

## Limitations

- No fixed random seed was recorded for either run.
- No dataset version or split information is available.
- No validation loss curves or held-out metrics are recorded.
- No generation quality evaluation was performed.
- Checkpoint files include optimizer state (~42 MB each) and are not suitable for inference.

## External Storage

Git LFS is configured to track model weight files under `model_weights/`. When pushed to GitHub or another Git LFS-compatible remote, the weight files will be stored in the LFS backend.

Current status: no externally verified weight release is published.