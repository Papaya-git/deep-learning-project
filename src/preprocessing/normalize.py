# -*- coding: utf-8 -*-
"""Image normalization preprocessing.

This module provides functions for normalizing image pixel values
using various normalization methods.
"""

import tensorflow as tf


def normalize_image(image, method='standard'):
    """Normalize pixel values of an image.
    
    Applies one of several normalization methods to the input image.
    
    Args:
        image: Input image tensor.
        method: Normalization method ('standard', 'centered', 'minmax').
        
    Returns:
        tf.Tensor: Normalized image tensor.
        
    Raises:
        ValueError: If an unknown normalization method is specified.
    """
    if method == 'standard':
        # Simple [0,1] normalization
        return image / 255.0
    
    elif method == 'centered':
        # Centered around 0 with std dev of 1
        # Common for models like EfficientNet
        image = image / 255.0
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        return (image - mean) / std
    
    elif method == 'minmax':
        # Scale to [-1, 1] range
        return (image / 127.5) - 1.0
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def apply_normalization(dataset, method='standard'):
    """Apply normalization to all images in a dataset.
    
    Args:
        dataset: TensorFlow dataset containing images.
        method: Normalization method ('standard', 'centered', 'minmax').
        
    Returns:
        tf.data.Dataset: Dataset with normalized images.
    """
    return dataset.map(
        lambda x, y: (normalize_image(x, method), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )