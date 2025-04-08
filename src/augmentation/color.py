# -*- coding: utf-8 -*-
"""Color augmentations for images.

This module provides functions for applying various color-based
augmentations to images, such as brightness, contrast, saturation,
and hue adjustments.
"""

import tensorflow as tf


def random_brightness(image, max_delta=0.2):
    """Randomly adjust image brightness.
    
    Args:
        image: Input image tensor.
        max_delta: Maximum brightness adjustment.
        
    Returns:
        tf.Tensor: Augmented image tensor.
    """
    return tf.image.random_brightness(image, max_delta)


def random_contrast(image, lower=0.8, upper=1.2):
    """Randomly adjust image contrast.
    
    Args:
        image: Input image tensor.
        lower: Lower bound for contrast adjustment.
        upper: Upper bound for contrast adjustment.
        
    Returns:
        tf.Tensor: Augmented image tensor.
    """
    return tf.image.random_contrast(image, lower, upper)


def random_saturation(image, lower=0.8, upper=1.2):
    """Randomly adjust image saturation.
    
    Args:
        image: Input image tensor.
        lower: Lower bound for saturation adjustment.
        upper: Upper bound for saturation adjustment.
        
    Returns:
        tf.Tensor: Augmented image tensor.
    """
    return tf.image.random_saturation(image, lower, upper)


def random_hue(image, max_delta=0.1):
    """Randomly adjust image hue.
    
    Args:
        image: Input image tensor.
        max_delta: Maximum hue adjustment (must be in [0, 0.5]).
        
    Returns:
        tf.Tensor: Augmented image tensor.
    """
    return tf.image.random_hue(image, max_delta)