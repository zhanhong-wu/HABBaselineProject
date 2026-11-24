# HAB Baseline Project

用于 Sentinel-2 藻华分割基线模型的训练、评估、预测导出和参数加载。

## 1. 安装环境

建议使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 准备数据

默认从项目上一级的 `Dataset` 目录读取数据：

```text
Dataset/
├── Train/
│   ├── Image/
│   └── Label/
├── Val/
│   ├── Image/
│   └── Label/
└── Test/
    ├── Image/
    └── Label/
```

影像与标签按文件名配对。例如影像 `example.tif` 对应标签 `example_label.tif`。默认输入为 12 通道影像，像元值缩放系数为 10000，切片大小为 256×256。

训练集归一化统计量默认读取项目上一级的 `dataset_stats_patch256_scale10000.json`。文件不存在时，程序会使用训练集计算并保存统计量。

## 3. 修改配置

训练、数据和推理参数集中在 `configs/default.json` 中，包括：

- `data_root`：数据集目录。
- `output_root`：结果输出目录，默认为 `outputs`。
- `stats_path`：训练集归一化统计量文件。
- `models`：需要运行的模型，设为 `["all"]` 可运行全部 15 个基线模型。
- `batch_size`、`num_epochs`、`learning_rate`、`num_workers`：训练参数。
- `target_test_images`：需要导出整幅预测结果的 Test 影像清单。

## 4. 训练与评估

运行全部基线模型：

```bash
python train.py --config configs/default.json --models all
```

只运行指定模型：

```bash
python train.py --models UPerNet_resnet34 Swin_Unet_tiny
```

常用参数：

```bash
python train.py \
  --data-root /path/to/Dataset \
  --output-root outputs \
  --stats-path /path/to/dataset_stats_patch256_scale10000.json \
  --epochs 150 \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda
```

每个模型的结果保存在 `outputs/<模型名>/`：

- `best_model.pth`：验证集综合得分最高的模型权重。
- `checkpoint.pth`：用于恢复训练的模型和优化器状态。
- `runs/`：TensorBoard 日志。
- `predictions.png`：测试样本可视化。
- `full_blocks/`：目标测试影像的整幅 PNG 和 GeoJSON 预测结果。

程序默认从 `outputs/<模型名>/checkpoint.pth` 恢复训练。禁止恢复训练：

```bash
python train.py --models Unet_resnet34 --no-resume
```

只评估已有的 `outputs/<模型名>/best_model.pth`：

```bash
python train.py --models Unet_resnet34 --eval-only
```

不生成预测可视化和整幅导出结果：

```bash
python train.py --models Unet_resnet34 --eval-only --skip-inference
```

## 5. 加载已有参数文件

`verify_checkpoints.py` 可检查已有模型权重和训练 checkpoint 是否能够严格加载。

参数文件应位于同一目录，并按以下格式命名：

```text
best_model_patch256_<模型名>_<tag>.pth
checkpoint_patch256_<模型名>_<tag>.pth
```

SMP 模型的 `tag` 为 `from_scratch`；Swin、ConvNeXt、MaxViT 和 DPT 的 `tag` 为 `custom_unet`。

验证全部模型：

```bash
python verify_checkpoints.py \
  --root /path/to/checkpoints \
  --models all \
  --mode both \
  --report checkpoint_verification.json
```

只验证一个模型：

```bash
python verify_checkpoints.py \
  --root /path/to/checkpoints \
  --models Unet_resnet34 \
  --mode both
```

只加载一种参数文件时，将 `--mode` 设为 `best` 或 `checkpoint`。加入 `--forward` 可额外执行一次 CPU 前向推理：

```bash
python verify_checkpoints.py \
  --root /path/to/checkpoints \
  --models Unet_resnet34 \
  --mode best \
  --forward
```

## 6. 运行单元测试

```bash
python -m unittest discover -s tests -v
```
