"""
Single-Image Prediction Interface
====================================
Production-ready interface for classifying individual hand gesture images
as Rock, Paper, or Scissors.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

from src.config import (
    CLASS_NAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    MODEL_DIR,
)


class RPSClassifier:
    """
    Rock-Paper-Scissors gesture classifier.

    Provides a clean interface for loading the trained model and
    making predictions on new images.

    Usage:
        classifier = RPSClassifier()
        result = classifier.predict("path/to/image.png")
        print(result)
    """

    def __init__(self, model_path=None):
        """
        Initialize the classifier by loading the trained model.

        Args:
            model_path (str, optional): Path to the model file.
                Defaults to the best model in the models directory.
        """
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "best_model.keras")
            if not os.path.exists(model_path):
                model_path = os.path.join(MODEL_DIR, "final_model.keras")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                "Please train the model first by running: python main.py"
            )

        self.model = tf.keras.models.load_model(model_path)
        self.class_names = CLASS_NAMES
        print(f"[RPSClassifier] Model loaded from: {model_path}")

    def preprocess_image(self, image_input):
        """
        Preprocess an image for prediction.

        Args:
            image_input: Can be a file path (str), PIL Image, or numpy array.

        Returns:
            np.ndarray: Preprocessed image ready for model input.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # If already preprocessed (in [-1, 1] range)
            if image_input.ndim == 3:
                image_input = np.expand_dims(image_input, axis=0)
            return image_input
        else:
            raise TypeError(
                f"Unsupported input type: {type(image_input)}. "
                "Expected str (path), PIL.Image, or numpy array."
            )

        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)  # [0, 255]
        arr = preprocess_input(arr)  # [-1, 1] for MobileNetV2
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self, image_input):
        """
        Classify a hand gesture image.

        Args:
            image_input: File path, PIL Image, or numpy array.

        Returns:
            dict: Prediction results containing:
                - 'label': Predicted class name (str)
                - 'confidence': Confidence score (float)
                - 'probabilities': Dict mapping each class to its probability
        """
        processed = self.preprocess_image(image_input)
        probabilities = self.model.predict(processed, verbose=0)[0]

        predicted_idx = np.argmax(probabilities)
        predicted_label = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        prob_dict = {
            name: float(prob)
            for name, prob in zip(self.class_names, probabilities)
        }

        return {
            "label": predicted_label,
            "confidence": confidence,
            "probabilities": prob_dict,
        }

    def predict_batch(self, image_inputs):
        """
        Classify a batch of images.

        Args:
            image_inputs: List of file paths, PIL Images, or numpy arrays.

        Returns:
            list[dict]: List of prediction results.
        """
        return [self.predict(img) for img in image_inputs]


def predict_image(image_path, model_path=None):
    """
    Convenience function for single image prediction.

    Args:
        image_path (str): Path to image file.
        model_path (str, optional): Path to model file.

    Returns:
        dict: Prediction results.
    """
    classifier = RPSClassifier(model_path=model_path)
    result = classifier.predict(image_path)

    print(f"\n{'=' * 50}")
    print(f"  Image:      {os.path.basename(image_path)}")
    print(f"  Prediction: {result['label'].upper()}")
    print(f"  Confidence: {result['confidence'] * 100:.2f}%")
    print(f"  Probabilities:")
    for name, prob in result['probabilities'].items():
        bar = "█" * int(prob * 30)
        print(f"    {name:<10} {prob * 100:6.2f}% {bar}")
    print(f"{'=' * 50}\n")

    return result
