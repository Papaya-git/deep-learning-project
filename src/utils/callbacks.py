# -*- coding: utf-8 -*-
"""Training callbacks"""

import os
from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau, 
    TensorBoard
)


def get_callbacks(config):
    """
    Create callbacks for model training
    
    Args:
        config: Configuration object with training parameters
        
    Returns:
        List of Keras callbacks and the model checkpoint path
    """
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = f'checkpoints/model_{timestamp}'
    
    # Create symlink to latest checkpoint
    latest_path = 'checkpoints/model_latest'
    if os.path.islink(latest_path):
        os.unlink(latest_path)
    os.symlink(f'model_{timestamp}', latest_path)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=config.checkpoints.monitor,
            save_best_only=config.checkpoints.save_best_only,
            save_weights_only=config.checkpoints.save_weights_only,
            save_format=config.checkpoints.save_format,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(log_dir=f'logs/tensorboard_{timestamp}')
    ]
    
    # Add early stopping if enabled
    if config.train.early_stopping.activate:
        callbacks.append(EarlyStopping(
            monitor=config.train.early_stopping.monitor,
            patience=config.train.early_stopping.patience,
            min_delta=config.train.early_stopping.min_delta,
            verbose=1
        ))
    
    # Add learning rate reduction if enabled
    if config.train.reduce_lr.activate:
        callbacks.append(ReduceLROnPlateau(
            monitor=config.train.reduce_lr.monitor,
            factor=config.train.reduce_lr.factor,
            patience=config.train.reduce_lr.patience,
            min_lr=config.train.reduce_lr.min_learning_rate,
            verbose=1
        ))
    
    return callbacks, checkpoint_path