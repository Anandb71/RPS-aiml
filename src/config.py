"""
Centralized Configuration & Hyperparameters
============================================
All tunable parameters for the RPS classification system.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ─── Class Labels ────────────────────────────────────────────────────────────
CLASS_NAMES = ["paper", "rock", "scissors"]
NUM_CLASSES = len(CLASS_NAMES)

# ─── Image Parameters ───────────────────────────────────────────────────────
IMG_HEIGHT = 150
IMG_WIDTH = 150
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ─── Data Split ──────────────────────────────────────────────────────────────
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15  # Held out from the full dataset before train/val split
RANDOM_SEED = 42

# ─── Training Hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# ─── Data Augmentation ──────────────────────────────────────────────────────
AUGMENTATION_CONFIG = {
    "rotation_range": 25,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.15,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest",
}

# ─── Ensure directories exist ───────────────────────────────────────────────
for _dir in [MODEL_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
