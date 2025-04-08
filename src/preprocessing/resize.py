# -*- coding: utf-8 -*-
"""Image resizing preprocessing.

This module provides functions for resizing images to a target size.
"""

import tensorflow as tf


def resize_image(image, target_size):
    """Resize an image to target dimensions.
    
    Args:
        image: Input image tensor.
        target_size: Tuple of (height, width) for target size.
        
    Returns:
        tf.Tensor: Resized image tensor.
    """
    return tf.image.resize(image, target_size)


def apply_resize(dataset, target_size):
    """Apply resizing to all images in a dataset.
    
    Args:
        dataset: TensorFlow dataset containing images.
        target_size: Tuple of (height, width) for target size.
        
    Returns:
        tf.data.Dataset: Dataset with resized images.
    """
    return dataset.map(
        lambda x, y: (resize_image(x, target_size), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )