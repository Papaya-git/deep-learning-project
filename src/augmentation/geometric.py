# -*- coding: utf-8 -*-
"""Geometric augmentations for images"""

import tensorflow as tf


def random_flip(image, horizontal=True, vertical=False):
    """
    Randomly flip an image horizontally and/or vertically
    
    Args:
        image: Input image tensor
        horizontal: Whether to apply horizontal flipping
        vertical: Whether to apply vertical flipping
        
    Returns:
        Augmented image tensor
    """
    if horizontal:
        image = tf.image.random_flip_left_right(image)
    
    if vertical:
        image = tf.image.random_flip_up_down(image)
    
    return image


def random_rotation(image, max_angle=0.2):
    """
    Randomly rotate an image
    
    Args:
        image: Input image tensor
        max_angle: Maximum rotation angle in radians
        
    Returns:
        Augmented image tensor
    """
    # Convert to radians and generate random angle
    angle = tf.random.uniform([], -max_angle, max_angle)
    
    # Apply rotation
    return tf.image.rot90(image, k=tf.cast(angle / (3.14159/2), tf.int32))


def random_shift(image, width_factor=0.2, height_factor=0.2):
    """
    Randomly shift an image horizontally and vertically
    
    Args:
        image: Input image tensor
        width_factor: Maximum horizontal shift as fraction of width
        height_factor: Maximum vertical shift as fraction of height
        
    Returns:
        Augmented image tensor
    """
    return tf.image.random_crop(
        tf.pad(image, [[0, 0], [height_factor, height_factor], [width_factor, width_factor], [0, 0]]),
        tf.shape(image)
    )


def random_zoom(image, zoom_range=(0.8, 1.2)):
    """
    Randomly zoom in or out on an image
    
    Args:
        image: Input image tensor
        zoom_range: Tuple of (min_zoom, max_zoom)
        
    Returns:
        Augmented image tensor
    """
    # Get image shape
    h, w, c = image.shape
    
    # Generate random zoom factor
    zoom = tf.random.uniform([], zoom_range[0], zoom_range[1])
    
    # Calculate new dimensions
    new_h = tf.cast(tf.cast(h, tf.float32) * zoom, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * zoom, tf.int32)
    
    # Resize and crop back to original size
    resized = tf.image.resize(image, [new_h, new_w])
    return tf.image.resize_with_crop_or_pad(resized, h, w) 