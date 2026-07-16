# Traffic Sign Recognition (GTSRB)

A research-grade PyTorch classification and evaluation pipeline on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

---

## 📊 Leaderboard (Official unseen test set - 12,630 images)

| Experiment | Architecture | Params | Image Size | EMA | Official Test Acc | Errors |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Exp001** | Custom CNN v2 | ~5.0M | 64x64 | ✅ | **`99.53%`** | **`59`** |
| **Exp001** | Custom CNN v2 (Raw) | ~5.0M | 64x64 | ❌ | **`99.46%`** | **`68`** |
| **Exp002** | EfficientNet-B0 | ~4.1M | 128x128 | ❌ | **`99.39%`** | **`77`** |
| **Exp003** | EfficientNet-B0 (Raw) | ~4.1M | 128x128 | ❌ | **`99.39%`** | **`77`** |
| **Exp003** | EfficientNet-B0 + EMA | ~4.1M | 128x128 | ✅ | **`99.36%`** | **`81`** |
| **Exp004** | ConvNeXt-Tiny | ~27.8M | 128x128 | *TBD* | *—* | *—* |
| **Exp005** | ResNet50 | ~23.5M | 128x128 | *TBD* | *—* | *—* |

---

## 🚀 How to Use

### 1. Training a model
Specify the yaml configuration path corresponding to the architecture you wish to optimize:
```bash
python train.py --config configs/models/cnn_v2.yaml
```

### 2. Evaluating a model on the official test set
Specify the config path and checkpoint location:
```bash
python evaluate.py --config configs/models/cnn_v2.yaml --checkpoint experiments/exp001_baseline_cnn/best_model.pth --use_ema
```
