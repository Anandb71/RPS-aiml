# Rock-Paper-Scissors Visual Recognition System

A production-grade deep learning system for identifying and classifying hand gestures (Rock, Paper, Scissors) from image data. Built with TensorFlow/Keras using MobileNetV2 transfer learning, achieving **100% test accuracy** on a held-out test set.

## Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **100.00%** |
| Test Loss | 0.0410 |
| Precision (all classes) | 1.0000 |
| Recall (all classes) | 1.0000 |
| F1-Score (all classes) | 1.0000 |

### Confusion Matrix (329 test samples)

|  | Predicted Paper | Predicted Rock | Predicted Scissors |
|--|:-:|:-:|:-:|
| **Paper** | 107 | 0 | 0 |
| **Rock** | 0 | 109 | 0 |
| **Scissors** | 0 | 0 | 113 |

Zero misclassifications across all gesture classes.

## Project Structure

```
RPS-aiml/
├── main.py                    # Entry point - full pipeline orchestration
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralized configuration & hyperparameters
│   ├── data_loader.py         # Data ingestion, augmentation & pipeline
│   ├── model.py               # MobileNetV2 transfer learning architecture
│   ├── train.py               # Two-phase training pipeline
│   ├── evaluate.py            # Model evaluation & metrics generation
│   ├── predict.py             # Single-image prediction interface (RPSClassifier)
│   └── visualize.py           # Publication-quality result visualization
├── data/                      # Dataset (not tracked in git)
│   ├── rock/                  # 726 images
│   ├── paper/                 # 712 images
│   └── scissors/              # 750 images
├── models/                    # Saved model artifacts
│   ├── best_model.keras       # Best checkpoint (by val_accuracy)
│   └── final_model.keras      # Final epoch model
└── results/                   # Evaluation results & visualizations
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── per_class_metrics.png
    ├── sample_predictions.png
    ├── confidence_distribution.png
    ├── class_distribution.png
    ├── evaluation_results.json
    ├── training_history.json
    └── training_log.csv
```

## Architecture

**Two-Phase Transfer Learning with MobileNetV2:**

### Phase 1: Feature Extraction (Frozen Backbone)
- MobileNetV2 pretrained on ImageNet with frozen weights
- Only the classification head is trained
- Learning rate: 1e-3
- Result: 99.70% validation accuracy

### Phase 2: Fine-Tuning (Partial Backbone Unfreezing)
- Top 54 layers of MobileNetV2 unfrozen for domain adaptation
- Data augmentation applied (rotation, shift, shear, zoom, flip, brightness)
- Learning rate: 1e-4 (10x lower for stability)
- Result: 100.00% validation accuracy

### Model Summary

| Component | Details |
|-----------|---------|
| Backbone | MobileNetV2 (ImageNet pretrained) |
| Pooling | Global Average Pooling (1280-d features) |
| Head | Dense(256)→BN→ReLU→Drop(0.5)→Dense(64)→BN→ReLU→Drop(0.3)→Softmax(3) |
| Total Parameters | 2.6M (345K trainable in Phase 1) |
| Regularization | L2 weight decay, dropout, batch normalization |

## Features

- **Transfer Learning**: MobileNetV2 backbone for robust feature extraction
- **Two-Phase Training**: Feature extraction then domain-specific fine-tuning
- **Data Augmentation**: Rotation, shift, shear, zoom, flip, brightness variation
- **Stratified Splitting**: Balanced train/validation/test splits (70/15/15)
- **Class Weighting**: Automatic handling of class imbalance
- **Training Callbacks**: Early stopping, LR scheduling, model checkpointing
- **Comprehensive Evaluation**: Per-class precision/recall/F1, confusion matrix
- **Production API**: `RPSClassifier` class for easy integration
- **6 Visualization Plots**: Training curves, confusion matrix, per-class metrics, sample predictions, confidence distribution, class distribution

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Full Pipeline (Train + Evaluate + Visualize)
```bash
python main.py
```

### Evaluate Existing Model
```bash
python main.py --evaluate
```

### Predict Single Image
```bash
python main.py --predict path/to/image.png
```

### Programmatic Usage
```python
from src.predict import RPSClassifier

classifier = RPSClassifier()
result = classifier.predict("path/to/hand_gesture.png")

print(f"Gesture: {result['label']}")          # "rock", "paper", or "scissors"
print(f"Confidence: {result['confidence']:.2%}")  # e.g., "99.87%"
print(f"Probabilities: {result['probabilities']}")
```

## Dataset

- **Source**: Julien de la Bruère-Terreault (CC-BY-SA 4.0)
- **Total Images**: 2,188 (Rock: 726, Paper: 712, Scissors: 750)
- **Format**: 300×200 RGB PNG images on green background
- **Preprocessing**: Resized to 150×150, MobileNetV2 normalization ([-1, 1])
- **Split**: 70% Train (1530) / 15% Validation (329) / 15% Test (329)

## License

Dataset: CC-BY-SA 4.0
