"""
CNN Model Architecture
========================
Transfer learning with MobileNetV2 backbone for Rock-Paper-Scissors
classification. Input images are expected in [-1, 1] range (preprocessed
externally).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers  # type: ignore

from src.config import IMG_SHAPE, NUM_CLASSES


def build_model():
    """
    Build a transfer learning model using MobileNetV2 as feature extractor
    with a custom classification head.

    Expects input images already preprocessed to [-1, 1] range.

    Architecture:
        - MobileNetV2 backbone (pretrained on ImageNet, frozen initially)
        - Global Average Pooling (1280-d feature vector)
        - Dense(256) -> BatchNorm -> ReLU -> Dropout(0.5)
        - Dense(64) -> BatchNorm -> ReLU -> Dropout(0.3)
        - Dense(3, softmax) output

    Returns:
        model: Keras Model.
        base_model: The MobileNetV2 backbone for fine-tuning control.
    """
    # Load MobileNetV2 pretrained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
    )

    # Freeze backbone for feature extraction phase
    base_model.trainable = False

    # Build model - NO preprocessing inside the graph
    # (preprocessing is done in data_loader.py)
    inputs = layers.Input(shape=IMG_SHAPE, name="input_image")

    # Pass through frozen backbone
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Classification head
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(0.5, name="fc1_dropout")(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4), name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Activation("relu", name="fc2_relu")(x)
    x = layers.Dropout(0.3, name="fc2_dropout")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = models.Model(inputs, outputs, name="RPS_MobileNetV2_Classifier")

    return model, base_model


def unfreeze_for_fine_tuning(model, base_model, fine_tune_at=100):
    """
    Unfreeze the top layers of the base model for fine-tuning.
    MobileNetV2 has 154 layers. Unfreezing from layer 100 gives ~54 trainable layers.

    Args:
        model: The full model.
        base_model: The MobileNetV2 backbone.
        fine_tune_at: Layer index from which to start unfreezing.

    Returns:
        model: Model with partially unfrozen backbone.
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    frozen = sum(1 for l in base_model.layers if not l.trainable)
    print(f"[Model] Fine-tuning: {trainable} trainable, {frozen} frozen layers")

    return model


def get_model_summary(model):
    """Return string representation of model architecture."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
