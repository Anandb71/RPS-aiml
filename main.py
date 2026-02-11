"""
Rock-Paper-Scissors Visual Recognition System
================================================
Main entry point orchestrating the complete pipeline:
    1. Data loading & preprocessing
    2. Two-phase training (feature extraction + fine-tuning)
    3. Evaluation on held-out test set
    4. Comprehensive visualization & reporting

Usage:
    python main.py                        # Full pipeline
    python main.py --evaluate             # Evaluate existing model
    python main.py --predict <image_path> # Predict single image

Author: Anand B
"""

import os
import sys
import json
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.config import RESULTS_DIR, MODEL_DIR
from src.train import train
from src.evaluate import evaluate_model, load_best_model
from src.predict import predict_image, RPSClassifier
from src.visualize import generate_all_visualizations
from src.data_loader import load_dataset


def run_full_pipeline():
    """Execute the complete training, evaluation, and visualization pipeline."""
    print("\n" + "=" * 70)
    print("  ROCK-PAPER-SCISSORS VISUAL RECOGNITION SYSTEM")
    print("  Full Pipeline Execution")
    print("=" * 70 + "\n")

    # Training (Phase 1: feature extraction + Phase 2: fine-tuning)
    model, history_dict, (X_test, y_test) = train()

    # Evaluation on held-out test set
    print("\n")
    eval_results = evaluate_model(model, X_test, y_test)

    # Reload split labels for distribution plot
    _, y_train, _, y_val, _, _ = load_dataset()

    # Generate all visualizations
    generate_all_visualizations(
        history_dict=history_dict,
        eval_results=eval_results,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
    )

    # Final Summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Test Accuracy:  {eval_results['test_accuracy'] * 100:.2f}%")
    print(f"  Test Loss:      {eval_results['test_loss']:.4f}")
    print(f"\n  Artifacts saved to:")
    print(f"    Models:  {MODEL_DIR}")
    print(f"    Results: {RESULTS_DIR}")
    print(f"\n  Files generated:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        print(f"    - {f} ({size / 1024:.1f} KB)")
    print()


def run_evaluate_only():
    """Evaluate an existing trained model."""
    print("\n" + "=" * 70)
    print("  EVALUATION MODE")
    print("=" * 70 + "\n")

    model = load_best_model()
    _, _, _, _, X_test, y_test = load_dataset()
    eval_results = evaluate_model(model, X_test, y_test)

    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history_dict = json.load(f)
        _, y_train, _, y_val, _, _ = load_dataset()
        generate_all_visualizations(
            history_dict=history_dict,
            eval_results=eval_results,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
        )


def run_predict(image_path):
    """Predict the gesture in a single image."""
    return predict_image(image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Rock-Paper-Scissors Visual Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Full pipeline
  python main.py --evaluate         # Evaluate existing model
  python main.py --predict img.png  # Predict single image
        """,
    )
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation only (requires trained model)")
    parser.add_argument("--predict", type=str, metavar="IMAGE_PATH",
                        help="Predict gesture in a single image")

    args = parser.parse_args()

    if args.predict:
        run_predict(args.predict)
    elif args.evaluate:
        run_evaluate_only()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
