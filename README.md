# Rock-Paper-Scissors Visual Recognition System

An image classification system that identifies hand gestures (Rock, Paper, Scissors) using deep learning. Built with TensorFlow/Keras and MobileNetV2 transfer learning.

## Results

- **Test Accuracy: 100%** on 329 unseen images
- Test Loss: 0.0410
- Precision / Recall / F1: 1.00 across all classes
- Zero misclassifications

### Confusion Matrix

|  | Predicted Paper | Predicted Rock | Predicted Scissors |
|--|:-:|:-:|:-:|
| **Paper** | 107 | 0 | 0 |
| **Rock** | 0 | 109 | 0 |
| **Scissors** | 0 | 0 | 113 |

## Project Structure

```
RPS-aiml/
├── main.py                 # Entry point for training, evaluation, prediction
├── requirements.txt
├── src/
│   ├── config.py           # Settings and hyperparameters
│   ├── data_loader.py      # Loads images, splits data, applies augmentation
│   ├── model.py            # MobileNetV2 transfer learning model
│   ├── train.py            # Two-phase training (feature extraction + fine-tuning)
│   ├── evaluate.py         # Evaluation metrics and classification report
│   ├── predict.py          # Predict gesture from a single image
│   └── visualize.py        # Generates plots and charts
├── data/                   # Dataset (rock/, paper/, scissors/)
├── models/                 # Saved trained models (.keras)
└── results/                # Charts, metrics, training logs
```

## How It Works

The model uses **MobileNetV2** (pretrained on ImageNet) as a feature extractor with a custom classification head on top.

**Phase 1 — Feature Extraction:** The MobileNetV2 backbone is frozen and only the classification head trains. This gets us to ~99.7% accuracy quickly.

**Phase 2 — Fine-Tuning:** The top layers of MobileNetV2 are unfrozen and the whole model trains together with a lower learning rate. This pushes accuracy to 100%.

Data augmentation (rotation, zoom, flips, brightness shifts) is applied during fine-tuning to make the model more robust.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train + Evaluate + Generate Visualizations
```bash
python main.py
```

### Evaluate an Already Trained Model
```bash
python main.py --evaluate
```

### Predict a Single Image
```bash
python main.py --predict path/to/image.png
```

### Use in Code
```python
from src.predict import RPSClassifier

classifier = RPSClassifier()
result = classifier.predict("hand_gesture.png")

print(result['label'])        # "rock", "paper", or "scissors"
print(result['confidence'])   # e.g. 0.9998
```

## Dataset

- **Source**: Julien de la Bruère-Terreault (CC-BY-SA 4.0)
- **Images**: 2,188 total — Rock (726), Paper (712), Scissors (750)
- **Format**: 300×200 RGB PNG on green background
- **Split**: 70% train / 15% validation / 15% test (stratified)

## License

Dataset: CC-BY-SA 4.0
