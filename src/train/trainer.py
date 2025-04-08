# -*- coding: utf-8 -*-
"""Model training implementation for binary classification"""

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping


class Trainer:
    """Class to handle model training for binary classification"""
    
    def __init__(self, model, config, callbacks, train_dataset, validation_dataset):
        """Initialize trainer with model and datasets"""
        self.model = model
        self.config = config
        self.callbacks = callbacks
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        
        # Compile model with optimizer, loss, and metrics
        self._compile_model()
    
    def _compile_model(self):
        """Compile the model with appropriate optimizer and loss for binary classification"""
        # Configure optimizer
        if self.config.train.optimizer.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.train.learning_rate)
        elif self.config.train.optimizer.lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.train.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.train.optimizer}")
        
        # Configure metrics with explicit binary settings
        metrics = [
            tf.keras.metrics.BinaryAccuracy(
                name='accuracy',
                threshold=self.config.metrics.classification_threshold
            ),
            Precision(
                name='precision',
                thresholds=[self.config.metrics.classification_threshold]
            ),
            Recall(
                name='recall',
                thresholds=[self.config.metrics.classification_threshold]
            ),
            AUC(
                name='auc',
                curve=self.config.metrics.auc_curve_type
            )
        ]
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=self.config.train.loss,
            metrics=metrics
        )
    
    def train(self):
        """Train the model"""
        history = self.model.fit(
            self.train_dataset,
            epochs=self.config.train.epochs,
            validation_data=self.validation_dataset,
            callbacks=self.callbacks,
            verbose=1
        )
        
        return history