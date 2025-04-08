# -*- coding: utf-8 -*-
"""Main entry point for training and evaluation.

This module serves as the main entry point for the binary image classification
system. It handles command-line argument parsing, model training, and evaluation.
"""

import argparse
import tensorflow as tf

from utils.config import Config
from data.data_loader import load_binary_dataset_with_split, get_class_names
from models.cnn import build_model
from utils.callbacks import get_callbacks
from train.trainer import Trainer
from evaluation.metrics import evaluate_model


def parse_args():
    """Parse command line arguments.
    
    Configures and parses command-line arguments for the application,
    including configuration path, execution mode, and model hyperparameters.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Binary Image Classification (Photo vs. Non-Photo)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the config file')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Override image size from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Override learning rate from config')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model for testing')
    return parser.parse_args()


def main():
    """Main function for training or testing a model.
    
    This function orchestrates the entire pipeline:
    1. Parses command-line arguments
    2. Loads configuration
    3. Prepares datasets
    4. Builds or loads the model
    5. Trains the model (if in training mode)
    6. Evaluates the model on the test set
    """
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    config.update_from_args(args)
    
    # Set memory growth for GPUs to prevent TensorFlow from allocating all GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Create datasets (augmentation is now handled by the model)
    train_dataset, validation_dataset, test_dataset = load_binary_dataset_with_split(config)
    
    # Get class names for evaluation
    class_names = get_class_names(config)
    print(f"Binary classification: {class_names}")
    
    if args.mode == 'train':
        # Build model with integrated augmentation
        model = build_model(config)
        
        # Get callbacks for model training
        callbacks, model_path = get_callbacks(config)
        
        # Create trainer and train model
        trainer = Trainer(model, config, callbacks, train_dataset, validation_dataset)
        history = trainer.train()
        
        # Evaluate on test set
        print("\nEvaluating on test set:")
        test_loss, test_acc = model.evaluate(test_dataset)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Generate detailed evaluation metrics
        metrics = evaluate_model(model, test_dataset, class_names)
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        # Print visualization paths
        print("\nVisualizations saved in:")
        print(" - reports/figures/confusion_matrix_*.png")
        print(" - reports/figures/roc_curve_*.png")
        print(" - reports/figures/precision_recall_*.png")
        
    elif args.mode == 'test':
        # Load model for testing
        model_path = args.model_path
        if not model_path:
            raise ValueError("Model path must be provided for test mode")
        
        model = tf.keras.models.load_model(model_path)
        
        # Fix dataset loading to get only test split
        _, _, test_dataset = load_binary_dataset_with_split(config)
        
        # Evaluate on test set
        print(f"Evaluating model from {model_path} on test set:")
        test_loss, test_acc = model.evaluate(test_dataset)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Generate detailed evaluation metrics
        metrics = evaluate_model(model, test_dataset, class_names)
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        # Print visualization paths
        print("\nVisualizations saved in:")
        print(" - reports/figures/confusion_matrix_*.png")
        print(" - reports/figures/roc_curve_*.png")
        print(" - reports/figures/precision_recall_*.png")
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()