# Flower Classification with Transfer Learning

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/YanFrenklakh/CNN)

Transfer learning project comparing VGG19, YOLOv5-cls, and EfficientNetV2B0 on the Oxford Flowers 102 dataset.

## Results Summary

| Model            | Test Accuracy | Parameters | Framework        |
| ---------------- | ------------- | ---------- | ---------------- |
| VGG19            | 92.38%        | 20.0M      | TensorFlow/Keras |
| YOLOv5-cls       | 90.82%        | ~7M        | PyTorch          |
| EfficientNetV2B0 | **93.95%**    | 6.1M       | TensorFlow/Keras |

**Best model:** EfficientNetV2B0 achieves highest accuracy with 3.3x fewer parameters than VGG19.

## Dataset

[Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

- 8,189 images
- 102 flower categories
- Stratified 50/25/25 split (train/val/test)

## Project Structure

```
├── notebooks/           # Jupyter notebooks (run in Google Colab)
│   ├── 01_data_pipeline.ipynb   # Data loading and preprocessing
│   ├── 02_vgg19_model.ipynb     # VGG19 transfer learning
│   ├── 03_yolov5_model.ipynb    # YOLOv5-cls (PyTorch)
│   └── 04_efficientnet_model.ipynb  # EfficientNetV2B0
├── outputs/             # Generated visualizations
└── README.md
```

## Running the Notebooks

1. Open notebooks in [Google Colab](https://colab.research.google.com/)
2. Runtime > Change runtime type > GPU
3. Run cells in order (01 → 02/03/04)

**Note:** Notebook 01 must be run first to create data splits.

## Requirements

- TensorFlow 2.x (Keras)
- PyTorch (for YOLOv5 notebook)
- scikit-learn
- matplotlib, seaborn

All dependencies are installed within the Colab notebooks.

## Key Techniques

- **Transfer Learning:** Pretrained ImageNet weights
- **Two-Phase Training:** Frozen backbone → Fine-tuning
- **Data Augmentation:** Horizontal flip, rotation, brightness/contrast
- **Class Balancing:** Weighted loss for imbalanced classes

## References

- [Oxford Flowers 102 Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [VGG19 Paper](https://arxiv.org/abs/1409.1556)
- [EfficientNetV2 Paper](https://arxiv.org/abs/2104.00298)
- [YOLOv5 Repository](https://github.com/ultralytics/yolov5)
