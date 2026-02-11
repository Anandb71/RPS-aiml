"""
Training Pipeline
===================
Two-phase training with MobileNetV2 transfer learning:
  Phase 1: Feature extraction (frozen backbone, train head only)
  Phase 2: Fine-tuning (unfreeze top backbone layers, low LR)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import (  # type: ignore
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
)

from src.config import (
    MODEL_DIR,
    RESULTS_DIR,
    LEARNING_RATE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LR,
    BATCH_SIZE,
)
from src.model import build_model, unfreeze_for_fine_tuning, get_model_summary
from src.data_loader import load_dataset, create_data_generators, get_class_weights


def _get_callbacks(phase="extraction"):
    """Configure training callbacks."""
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        CSVLogger(
            os.path.join(RESULTS_DIR, f"training_log_{phase}.csv"),
            separator=",",
            append=False,
        ),
    ]
    return callbacks


def train():
    """
    Execute two-phase training pipeline.

    Returns:
        model: Trained Keras model.
        history: Combined training history dict.
        test_data: Tuple (X_test, y_test) for evaluation.
    """
    # ─── Step 1: Load data ──────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Loading Dataset")
    print("=" * 70)
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # ─── Step 2: Compute class weights ──────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Preparing Training")
    print("=" * 70)
    class_weights = get_class_weights(y_train)

    # ─── Step 3: Build model ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Building Model Architecture")
    print("=" * 70)
    model, base_model = build_model()

    summary = get_model_summary(model)
    print(summary)
    with open(os.path.join(RESULTS_DIR, "model_architecture.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Feature Extraction (frozen backbone, train head only)
    # Train directly on numpy arrays - no generator issues
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: Feature Extraction (Frozen Backbone)")
    print("=" * 70)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    phase1_epochs = 20
    callbacks_p1 = _get_callbacks(phase="extraction")

    history_p1 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=phase1_epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_p1,
        class_weight=class_weights,
        verbose=1,
    )

    best_p1 = max(history_p1.history["val_accuracy"])
    print(f"\n[Phase 1] Best val_accuracy: {best_p1:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Fine-Tuning with data augmentation
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: Fine-Tuning (Unfreezing Top Backbone Layers)")
    print("=" * 70)

    model = unfreeze_for_fine_tuning(model, base_model, fine_tune_at=100)

    fine_tune_lr = LEARNING_RATE / 10
    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"[Phase 2] Recompiled with lr={fine_tune_lr}")

    # Use augmented data for fine-tuning
    train_gen, val_gen = create_data_generators(X_train, y_train, X_val, y_val)

    phase2_epochs = 30
    callbacks_p2 = _get_callbacks(phase="finetuning")

    history_p2 = model.fit(
        train_gen,
        epochs=phase2_epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_p2,
        class_weight=class_weights,
        verbose=1,
    )

    best_p2 = max(history_p2.history["val_accuracy"])
    print(f"\n[Phase 2] Best val_accuracy: {best_p2:.4f}")

    # ─── Combine histories ──────────────────────────────────────────────
    combined_history = {}
    for key in history_p1.history:
        combined_history[key] = (
            [float(v) for v in history_p1.history[key]] +
            [float(v) for v in history_p2.history[key]]
        )

    # ─── Save ───────────────────────────────────────────────────────────
    final_model_path = os.path.join(MODEL_DIR, "final_model.keras")
    model.save(final_model_path)
    print(f"\n[Trainer] Final model saved to: {final_model_path}")

    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"[Trainer] Training history saved to: {history_path}")

    return model, combined_history, (X_test, y_test)


if __name__ == "__main__":
    train()
