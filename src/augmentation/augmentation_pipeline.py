# -*- coding: utf-8 -*-
"""Custom augmentation layer using existing augmentation functions.

This module defines a custom Keras layer that applies various augmentation
techniques to images during training, using the existing augmentation functions.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from augmentation.geometric import random_flip, random_rotation, random_shift, random_zoom
from augmentation.color import random_brightness, random_contrast, random_saturation, random_hue


class CustomAugmentationLayer(Layer):
    """Custom layer that applies augmentation using existing augmentation functions.
    
    This layer only applies augmentation during training, not during inference.
    It supports different augmentation strategies with varying intensities.
    """
    
    def __init__(self, augmentation_type='default', **kwargs):
        """Initialize the augmentation layer.
        
        Args:
            augmentation_type: Type of augmentation to apply ('light', 'default', 'heavy').
            **kwargs: Additional arguments to pass to the base Layer class.
        """
        super(CustomAugmentationLayer, self).__init__(**kwargs)
        self.augmentation_type = augmentation_type
    
    def call(self, inputs, training=None):
        """Apply augmentation to inputs.
        
        This method is called when the layer is invoked. It applies the
        appropriate augmentation strategy if in training mode.
        
        Args:
            inputs: Input tensor (batch of images).
            training: Boolean indicating whether in training mode.
            
        Returns:
            tf.Tensor: Augmented tensor if in training mode, otherwise original tensor.
        """
        # Only apply augmentation during training
        if training:
            if self.augmentation_type == 'light':
                return self._apply_light_augmentation(inputs)
            elif self.augmentation_type == 'default':
                return self._apply_default_augmentation(inputs)
            elif self.augmentation_type == 'heavy':
                return self._apply_heavy_augmentation(inputs)
        
        # During inference, return inputs unchanged
        return inputs
    
    def _apply_light_augmentation(self, image):
        """Apply light augmentation using existing functions.
        
        Applies a minimal set of augmentations with conservative parameters.
        
        Args:
            image: Input image tensor.
            
        Returns:
            tf.Tensor: Augmented image tensor.
        """
        # Apply geometric transformations
        image = random_flip(image, horizontal=True, vertical=False)
        
        # Apply color transformations
        image = random_brightness(image, max_delta=0.1)
        image = random_contrast(image, lower=0.9, upper=1.1)
        
        return image
    
    def _apply_default_augmentation(self, image):
        """Apply default augmentation using existing functions.
        
        Applies a standard set of augmentations with moderate parameters.
        
        Args:
            image: Input image tensor.
            
        Returns:
            tf.Tensor: Augmented image tensor.
        """
        # Apply geometric transformations
        image = random_flip(image, horizontal=True, vertical=False)
        image = random_rotation(image, max_angle=0.1)
        image = random_shift(image, width_factor=0.1, height_factor=0.1)
        
        # Apply color transformations
        image = random_brightness(image, max_delta=0.1)
        image = random_contrast(image, lower=0.8, upper=1.2)
        
        return image
    
    def _apply_heavy_augmentation(self, image):
        """Apply heavy augmentation using existing functions.
        
        Applies an extensive set of augmentations with aggressive parameters.
        
        Args:
            image: Input image tensor.
            
        Returns:
            tf.Tensor: Augmented image tensor.
        """
        # Apply geometric transformations
        image = random_flip(image, horizontal=True, vertical=True)
        image = random_rotation(image, max_angle=0.2)
        image = random_shift(image, width_factor=0.2, height_factor=0.2)
        image = random_zoom(image, zoom_range=(0.8, 1.2))
        
        # Apply color transformations
        image = random_brightness(image, max_delta=0.2)
        image = random_contrast(image, lower=0.7, upper=1.3)
        image = random_saturation(image, lower=0.7, upper=1.3)
        image = random_hue(image, max_delta=0.1)
        
        return image
    
    def get_config(self):
        """Get layer configuration for serialization.
        
        Returns:
            dict: Layer configuration dictionary.
        """
        config = super(CustomAugmentationLayer, self).get_config()
        config.update({"augmentation_type": self.augmentation_type})
        return config