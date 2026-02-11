"""
Results Visualization & Reporting
====================================
Generates publication-quality visualizations of training progress,
model performance, and classification results.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.config import CLASS_NAMES, RESULTS_DIR


# ─── Plot Style Configuration ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
sns.set_palette("deep")


def plot_training_history(history_dict, save_dir=None):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history_dict (dict): Training history with keys 'accuracy', 'val_accuracy',
                             'loss', 'val_loss'.
        save_dir (str, optional): Directory to save plots. Defaults to RESULTS_DIR.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    epochs = range(1, len(history_dict["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ─── Accuracy Plot ──────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(epochs, history_dict["accuracy"], "b-o", markersize=4,
             label="Training Accuracy", linewidth=2)
    ax1.plot(epochs, history_dict["val_accuracy"], "r-s", markersize=4,
             label="Validation Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy Over Epochs", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right", fontsize=11)
    ax1.set_ylim([0, 1.05])

    # Mark best validation accuracy
    best_val_acc_idx = np.argmax(history_dict["val_accuracy"])
    best_val_acc = history_dict["val_accuracy"][best_val_acc_idx]
    ax1.axhline(y=best_val_acc, color="green", linestyle="--", alpha=0.5)
    ax1.annotate(
        f"Best: {best_val_acc:.4f} (epoch {best_val_acc_idx + 1})",
        xy=(best_val_acc_idx + 1, best_val_acc),
        fontsize=10, color="green", fontweight="bold",
        xytext=(10, -20), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="green"),
    )

    # ─── Loss Plot ──────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(epochs, history_dict["loss"], "b-o", markersize=4,
             label="Training Loss", linewidth=2)
    ax2.plot(epochs, history_dict["val_loss"], "r-s", markersize=4,
             label="Validation Loss", linewidth=2)
    ax2.set_title("Model Loss Over Epochs", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right", fontsize=11)

    # Mark best validation loss
    best_val_loss_idx = np.argmin(history_dict["val_loss"])
    best_val_loss = history_dict["val_loss"][best_val_loss_idx]
    ax2.axhline(y=best_val_loss, color="green", linestyle="--", alpha=0.5)
    ax2.annotate(
        f"Best: {best_val_loss:.4f} (epoch {best_val_loss_idx + 1})",
        xy=(best_val_loss_idx + 1, best_val_loss),
        fontsize=10, color="green", fontweight="bold",
        xytext=(10, 20), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="green"),
    )

    plt.tight_layout(pad=3.0)
    path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Training curves saved to: {path}")


def plot_confusion_matrix(y_true, y_pred, save_dir=None):
    """
    Plot a detailed confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        save_dir (str, optional): Directory to save plot.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ─── Raw Counts ─────────────────────────────────────────────────────
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[0], linewidths=1, linecolor="gray",
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # ─── Normalized (Percentages) ───────────────────────────────────────
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="RdYlGn",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[1], linewidths=1, linecolor="gray",
        vmin=0, vmax=1,
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes[1].set_title("Confusion Matrix (Normalized %)", fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout(pad=3.0)
    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Confusion matrix saved to: {path}")


def plot_per_class_metrics(eval_results, save_dir=None):
    """
    Plot per-class precision, recall, and F1-score as grouped bar charts.

    Args:
        eval_results (dict): Evaluation results with 'per_class_metrics'.
        save_dir (str, optional): Directory to save plot.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    metrics = eval_results["per_class_metrics"]

    classes = list(metrics.keys())
    precision = [metrics[c]["precision"] for c in classes]
    recall = [metrics[c]["recall"] for c in classes]
    f1 = [metrics[c]["f1_score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, precision, width, label="Precision",
                   color="#2196F3", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, recall, width, label="Recall",
                   color="#4CAF50", edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, f1, width, label="F1-Score",
                   color="#FF9800", edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.set_title("Per-Class Performance Metrics", fontweight="bold", fontsize=14)
    ax.set_xlabel("Gesture Class")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=12)
    ax.set_ylim([0, 1.15])
    ax.legend(loc="upper left", fontsize=11)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "per_class_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Per-class metrics saved to: {path}")


def plot_prediction_samples(X_test, y_true, y_pred, y_pred_probs,
                            num_samples=15, save_dir=None):
    """
    Display sample predictions with images, true/predicted labels,
    and confidence scores. Shows both correct and incorrect predictions.

    Args:
        X_test: Test images.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_pred_probs: Prediction probabilities.
        num_samples: Number of samples to display.
        save_dir: Directory to save plot.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    # Find correct and incorrect predictions
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask

    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]

    # Sample: prioritize showing some errors if they exist
    n_incorrect = min(len(incorrect_indices), max(3, num_samples // 3))
    n_correct = num_samples - n_incorrect

    selected = []
    if len(incorrect_indices) > 0:
        selected.extend(
            np.random.RandomState(42).choice(
                incorrect_indices, size=min(n_incorrect, len(incorrect_indices)),
                replace=False
            ).tolist()
        )
    if len(correct_indices) > 0:
        selected.extend(
            np.random.RandomState(42).choice(
                correct_indices, size=min(n_correct, len(correct_indices)),
                replace=False
            ).tolist()
        )

    np.random.RandomState(42).shuffle(selected)
    selected = selected[:num_samples]

    cols = 5
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows * cols == 1 else axes.flatten()

    for i, idx in enumerate(selected):
        ax = axes[i]
        # Convert from [-1, 1] to [0, 1] for display
        display_img = (X_test[idx] + 1.0) / 2.0
        display_img = np.clip(display_img, 0, 1)
        ax.imshow(display_img)
        ax.axis("off")

        true_label = CLASS_NAMES[y_true[idx]]
        pred_label = CLASS_NAMES[y_pred[idx]]
        confidence = y_pred_probs[idx][y_pred[idx]] * 100

        is_correct = y_true[idx] == y_pred[idx]
        color = "green" if is_correct else "red"
        symbol = "✓" if is_correct else "✗"

        ax.set_title(
            f"{symbol} Pred: {pred_label.capitalize()}\n"
            f"True: {true_label.capitalize()} ({confidence:.1f}%)",
            fontsize=10, color=color, fontweight="bold",
        )

    # Hide unused axes
    for j in range(len(selected), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Sample Predictions (Green=Correct, Red=Incorrect)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "sample_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Sample predictions saved to: {path}")


def plot_class_distribution(y_train, y_val, y_test, save_dir=None):
    """
    Plot the distribution of classes across train/val/test splits.

    Args:
        y_train, y_val, y_test: Labels for each split.
        save_dir: Directory to save plot.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    splits = {"Train": y_train, "Validation": y_val, "Test": y_test}
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, (split_name, labels) in enumerate(splits.items()):
        counts = [np.sum(labels == c) for c in range(len(CLASS_NAMES))]
        bars = ax.bar(x + i * width, counts, width, label=split_name,
                      color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, counts):
            ax.annotate(
                str(count), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold",
            )

    ax.set_title("Class Distribution Across Data Splits", fontweight="bold", fontsize=14)
    ax.set_xlabel("Gesture Class")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=12)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Class distribution saved to: {path}")


def plot_confidence_distribution(y_true, y_pred, y_pred_probs, save_dir=None):
    """
    Plot the distribution of prediction confidence for correct vs incorrect predictions.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_pred_probs: Prediction probabilities.
        save_dir: Directory to save plot.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    confidences = np.max(y_pred_probs, axis=1)
    correct_mask = y_true == y_pred

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(confidences[correct_mask], bins=30, alpha=0.7, label="Correct",
            color="#4CAF50", edgecolor="white")
    if np.sum(~correct_mask) > 0:
        ax.hist(confidences[~correct_mask], bins=30, alpha=0.7, label="Incorrect",
                color="#F44336", edgecolor="white")

    ax.set_title("Prediction Confidence Distribution", fontweight="bold", fontsize=14)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=11)
    ax.axvline(x=np.mean(confidences), color="blue", linestyle="--", alpha=0.7,
               label=f"Mean: {np.mean(confidences):.3f}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, "confidence_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Confidence distribution saved to: {path}")


def generate_all_visualizations(history_dict, eval_results, X_test,
                                 y_train=None, y_val=None):
    """
    Generate all visualization plots.

    Args:
        history_dict: Training history dictionary.
        eval_results: Evaluation results dictionary.
        X_test: Test images for sample predictions.
        y_train, y_val: Training and validation labels for distribution plot.
    """
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    y_true = np.array(eval_results["predictions"]["y_true"])
    y_pred = np.array(eval_results["predictions"]["y_pred"])
    y_pred_probs = np.array(eval_results["predictions"]["y_pred_probs"])

    # 1. Training curves
    plot_training_history(history_dict)

    # 2. Confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # 3. Per-class metrics
    plot_per_class_metrics(eval_results)

    # 4. Sample predictions
    plot_prediction_samples(X_test, y_true, y_pred, y_pred_probs)

    # 5. Confidence distribution
    plot_confidence_distribution(y_true, y_pred, y_pred_probs)

    # 6. Class distribution (if split labels provided)
    y_test = y_true
    if y_train is not None and y_val is not None:
        plot_class_distribution(y_train, y_val, y_test)

    print(f"\n[Visualizer] All visualizations saved to: {RESULTS_DIR}")
