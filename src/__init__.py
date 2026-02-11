"""
Rock-Paper-Scissors Visual Recognition System
================================================
A production-grade CNN-based image classification system for identifying
hand gestures representing Rock, Paper, and Scissors.

Project Structure:
    src/
        config.py        - Centralized configuration & hyperparameters
        data_loader.py   - Data ingestion, augmentation & pipeline
        model.py         - CNN architecture definition
        train.py         - Training loop with callbacks
        evaluate.py      - Model evaluation & metrics generation
        predict.py       - Single-image prediction interface
        visualize.py     - Result visualization & reporting
    main.py              - Entry point orchestrating the full pipeline

Author: Anand B
"""
