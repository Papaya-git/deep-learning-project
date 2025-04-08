# -*- coding: utf-8 -*-
"""Complete preprocessing pipeline.

This module provides functions for creating and applying a complete
preprocessing pipeline to images, including resizing and normalization.
"""

import tensorflow as tf
from preprocessing.resize import resize_image
from preprocessing.normalize import normalize_image
from tensorflow.keras import layers


def preprocess_image(image, target_size, normalization_method='standard'):
    """Apply full preprocessing pipeline to a single image.
    
    Applies a sequence of preprocessing operations to prepare an image
    for model input, including resizing and normalization.
    
    Args:
        image: Input image tensor.
        target_size: Tuple of (height, width) for target size.
        normalization_method: Method for pixel normalization.
        
    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    # Resize the image to target dimensions
    image = resize_image(image, target_size)
    
    # Normalize pixel values
    image = normalize_image(image, method=normalization_method)
    
    return image


def create_preprocessing_pipeline(config) -> tf.keras.Model:
    """Create a complete preprocessing pipeline with configurable parameters.
    
    Args:
        config: Configuration object with preprocessing parameters containing:
            - data.image_size: Target dimensions (height, width)
            - data.channels: Number of image channels
            - preprocessing.normalization.method: Normalization strategy
            - preprocessing.resize.activate: Whether to apply resizing
            - preprocessing.normalization.activate: Whether to apply normalization
            
    Returns:
        tf.keras.Sequential: Preprocessing pipeline model
    """
    layers_list = []
    
    # Add resizing if activated in config
    if config.preprocessing.resize.activate:
        target_size = tuple(config.data.image_size)
        layers_list.append(
            layers.Lambda(
                lambda x: resize_image(x, target_size),
                name="resize_layer"
            )
        )
    
    # Add normalization if activated in config
    if config.preprocessing.normalization.activate:
        normalization_method = config.preprocessing.normalization.method
        layers_list.append(
            layers.Lambda(
                lambda x: normalize_image(x, method=normalization_method),
                name="normalization_layer"
            )
        )
    
    pipeline = tf.keras.Sequential(layers_list)
    pipeline.build([None, None, None, config.data.channels])
    return pipeline


def create_inference_preprocessing(config):
    """Create preprocessing function for inference that preserves batch dimension.
    
    Args:
        config: Configuration object with preprocessing parameters
        
    Returns:
        Callable: Preprocessing function that can be applied to batches of data
    """
    pipeline = create_preprocessing_pipeline(config)
    return lambda x: pipeline(x, training=False)