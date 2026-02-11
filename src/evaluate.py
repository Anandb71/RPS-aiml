"""
Model Evaluation & Metrics Generation
========================================
Comprehensive evaluation of the trained model on the held-out test set,
generating classification reports, confusion matrices, and per-class metrics.
"""

import os
import json
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import tensorflow as tf

from src.config import CLASS_NAMES, MODEL_DIR, RESULTS_DIR


def evaluate_model(model, X_test, y_test):
    """
    Perform full evaluation of the model on the test set.

    Args:
        model: Trained Keras model.
        X_test: Test images array.
        y_test: Test labels array.

    Returns:
        results (dict): Comprehensive evaluation results.
    """
    print("=" * 70)
    print("MODEL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)

    # ─── Predictions ────────────────────────────────────────────────────
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # ─── Overall Metrics ────────────────────────────────────────────────
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    overall_accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # ─── Per-Class Metrics ──────────────────────────────────────────────
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(CLASS_NAMES)))
    )

    per_class_metrics = {}
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 54)
    for i, name in enumerate(CLASS_NAMES):
        per_class_metrics[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }
        print(
            f"  {name:<12} {precision[i]:>10.4f} {recall[i]:>10.4f} "
            f"{f1[i]:>10.4f} {support[i]:>10d}"
        )

    # ─── Classification Report ──────────────────────────────────────────
    report = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print(f"\n  Full Classification Report:\n{report}")

    # ─── Confusion Matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {cm}")

    # ─── Compile Results ────────────────────────────────────────────────
    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "overall_accuracy": float(overall_accuracy),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "predictions": {
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_pred_probs": y_pred_probs.tolist(),
        },
    }

    # Save results to JSON
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    save_results = {k: v for k, v in results.items() if k != "predictions"}
    save_results["num_test_samples"] = len(y_test)
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n[Evaluator] Results saved to: {results_path}")

    return results


def load_best_model():
    """
    Load the best saved model from disk.

    Returns:
        model: Loaded Keras model.
    """
    model_path = os.path.join(MODEL_DIR, "best_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "final_model.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found. Expected at: {model_path}"
        )

    model = tf.keras.models.load_model(model_path)
    print(f"[Evaluator] Model loaded from: {model_path}")
    return model
