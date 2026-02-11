"""
Data Loader & Preprocessing Pipeline
======================================
Handles dataset ingestion, stratified splitting, image augmentation,
and data preparation. Images are preprocessed to MobileNetV2 expected
range [-1, 1] using the official preprocess_input function.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
from PIL import Image

from src.config import (
    DATA_DIR,
    CLASS_NAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    BATCH_SIZE,
    AUGMENTATION_CONFIG,
)


def _collect_image_paths_and_labels():
    """
    Walk the data directory and collect all image file paths with their
    corresponding class labels.

    Returns:
        file_paths (list[str]): Absolute paths to each image.
        labels (list[int]): Integer class label for each image.
    """
    file_paths = []
    labels = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected class directory not found: {class_dir}"
            )

        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)

    print(f"[DataLoader] Collected {len(file_paths)} images across {len(CLASS_NAMES)} classes.")
    for idx, name in enumerate(CLASS_NAMES):
        count = labels.count(idx)
        print(f"  - {name}: {count} images")

    return file_paths, labels


def _load_and_preprocess_image(path):
    """
    Load a single image, resize to target dimensions, and preprocess
    for MobileNetV2 (scale to [-1, 1]).

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image of shape (IMG_HEIGHT, IMG_WIDTH, 3)
                    with pixel values in [-1, 1].
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)  # [0, 255]
    arr = preprocess_input(arr)  # [-1, 1] for MobileNetV2
    return arr


def load_dataset():
    """
    Load the full dataset into memory, preprocess, and perform stratified
    splitting into train, validation, and test sets.

    Returns:
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        X_test, y_test: Test data and labels (completely held out).
    """
    file_paths, labels = _collect_image_paths_and_labels()

    print("[DataLoader] Loading and preprocessing images...")
    images = np.array([_load_and_preprocess_image(p) for p in file_paths])
    labels = np.array(labels)

    # First split: separate out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    # Second split: split remaining into train and validation
    val_ratio = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"[DataLoader] Split complete:")
    print(f"  - Training:   {X_train.shape[0]} samples")
    print(f"  - Validation: {X_val.shape[0]} samples")
    print(f"  - Test:       {X_test.shape[0]} samples")
    print(f"  - Pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_data_generators(X_train, y_train, X_val, y_val):
    """
    Create Keras ImageDataGenerators with augmentation for training
    and no augmentation for validation.

    Note: Data is already preprocessed to [-1, 1] range.
    Augmentation operations (rotation, shift, etc.) preserve the range.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        train_gen: Augmented training data generator.
        val_gen: Validation data generator.
    """
    train_datagen = ImageDataGenerator(**AUGMENTATION_CONFIG)
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_gen = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"[DataLoader] Data generators created (batch_size={BATCH_SIZE}).")
    return train_gen, val_gen


def get_class_weights(y_train):
    """
    Compute class weights to handle any class imbalance.

    Args:
        y_train: Training labels.

    Returns:
        dict: Mapping from class index to weight.
    """
    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = {i: w for i, w in enumerate(weights)}
    print(f"[DataLoader] Class weights: {class_weights}")
    return class_weights
