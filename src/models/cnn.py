# -*- coding: utf-8 -*-
"""CNN model architecture for binary classification.

This module defines the CNN architecture used for binary image classification,
with integrated data augmentation capabilities.
"""

import tensorflow as tf
from tensorflow.keras import layers
from src.augmentation.augmentation_pipeline import CustomAugmentationLayer


def build_model(config, augmentation_type=None):
    """Build a CNN model for binary image classification with integrated data augmentation.
    
    Constructs a CNN model with the specified architecture and integrated
    data augmentation. The augmentation is only applied during training.
    
    Args:
        config: Configuration object with model parameters.
        augmentation_type: Type of augmentation to apply ('light', 'default', 'heavy', 'none').
        
    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
    input_shape = (*config.data.image_size, config.data.channels)
    
    # Start with an input layer
    inputs = layers.Input(shape=input_shape)
    
    # Add preprocessing pipeline (NEW INTEGRATION)
    if config.preprocessing.resize.activate or config.preprocessing.normalization.activate:
        from src.preprocessing.preprocessing_pipeline import create_preprocessing_pipeline
        x = create_preprocessing_pipeline(config)(inputs)
    else:
        x = inputs
    
    # Use config augmentation activation flag
    if config.augmentation.activate:
        augmentation_type = config.augmentation.type
    
    # Add custom augmentation layer (only applied during training)
    if augmentation_type != 'none':
        x = CustomAugmentationLayer(augmentation_type=augmentation_type)(x)
    
    # Build convolutional blocks using config parameters
    for filters in config.model.conv_filters:
        x = layers.Conv2D(
            filters, 
            (3, 3), 
            activation='relu', 
            padding='same',
            kernel_initializer=config.model.kernel_initializer
        )(x)
        if config.model.batch_norm.activate:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
    # Add dropout if configured
    if config.model.dropout.activate:
        for rate in config.model.dropout.rates:
            x = layers.Dropout(rate)(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(
        config.model.dense_units,
        activation='relu',
        kernel_initializer=config.model.kernel_initializer
    )(x)
    
    # Final classification layer
    outputs = layers.Dense(
        1, 
        activation=config.model.final_activation,
        kernel_initializer=config.model.kernel_initializer
    )(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Print model summary
    model.summary()
    
    return model