# -*- coding: utf-8 -*-
"""Configuration management using dataclasses"""

from dataclasses import dataclass
from typing import List, Any
import yaml
from pathlib import Path


@dataclass
class ShuffleConfig:
    """Shuffle configuration"""
    activate: bool


@dataclass
class StratifyConfig:
    """Stratification configuration"""
    activate: bool


@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_path: str
    original_classes: List[str]
    binary_classes: List[str]
    image_size: List[int]
    channels: int
    seed: int
    shuffle_buffer_size: int
    shuffle: ShuffleConfig


@dataclass
class NormalizationConfig:
    """Normalization configuration"""
    activate: bool
    method: str


@dataclass
class ResizeConfig:
    """Resize configuration"""
    activate: bool
    method: str


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing"""
    normalization: NormalizationConfig
    resize: ResizeConfig
    rescale_factor: float


@dataclass
class DropoutConfig:
    """Dropout configuration"""
    activate: bool
    rates: List[float]


@dataclass
class BatchNormConfig:
    """Batch normalization configuration"""
    activate: bool


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    conv_filters: List[int]
    dense_units: int
    dropout: DropoutConfig
    final_activation: str
    kernel_initializer: str
    batch_norm: BatchNormConfig


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration"""
    activate: bool
    patience: int
    min_delta: float
    monitor: str


@dataclass
class ReduceLRConfig:
    """Learning rate reduction configuration"""
    activate: bool
    factor: float
    patience: int
    min_learning_rate: float
    monitor: str


@dataclass
class TrainConfig:
    """Configuration for training parameters"""
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    loss: str
    metrics: List[str]
    early_stopping: EarlyStoppingConfig
    reduce_lr: ReduceLRConfig


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation"""
    activate: bool
    type: str
    rotation_range: int
    width_shift_range: float
    height_shift_range: float
    zoom_range: float
    horizontal_flip: bool
    fill_mode: str


@dataclass
class CheckpointsConfig:
    """Configuration for model checkpoints"""
    activate: bool
    monitor: str
    save_best_only: bool
    save_weights_only: bool
    save_format: str


@dataclass
class DataSplittingConfig:
    """Configuration for dataset splitting"""
    train_size: float
    validation_size: float
    test_size: float
    stratify: StratifyConfig
    random_state: int


@dataclass
class CacheConfig:
    """Cache configuration"""
    activate: bool


@dataclass
class TrainingProcessConfig:
    """Configuration for training process parameters"""
    prefetch_buffer_size: str
    cache: CacheConfig
    num_parallel_calls: str
    experimental_features: bool = False


@dataclass
class MetricsConfig:
    """Configuration for model metrics"""
    classification_threshold: float
    auc_curve_type: str


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig
    data_splitting: DataSplittingConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    train: TrainConfig
    augmentation: AugmentationConfig
    checkpoints: CheckpointsConfig
    metrics: MetricsConfig
    training_process: TrainingProcessConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Config object with loaded configuration
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        # Ensure nested dataclasses are properly initialized
        data_dict = config_dict['data']
        shuffle_config = ShuffleConfig(**data_dict.pop('shuffle'))
        
        data_splitting_dict = config_dict['data_splitting']
        stratify_config = StratifyConfig(**data_splitting_dict.pop('stratify'))
        
        preprocessing_dict = config_dict['preprocessing']
        normalization_config = NormalizationConfig(**preprocessing_dict.pop('normalization'))
        resize_config = ResizeConfig(**preprocessing_dict.pop('resize'))
        
        model_dict = config_dict['model']
        dropout_config = DropoutConfig(**model_dict.pop('dropout'))
        batch_norm_config = BatchNormConfig(**model_dict.pop('batch_norm'))
        
        train_dict = config_dict['train']
        early_stopping_config = EarlyStoppingConfig(**train_dict.pop('early_stopping'))
        reduce_lr_config = ReduceLRConfig(**train_dict.pop('reduce_lr'))
        
        training_process_dict = config_dict['training_process']
        cache_config = CacheConfig(**training_process_dict.pop('cache'))
        
        return cls(
            data=DataConfig(
                **data_dict,
                shuffle=shuffle_config
            ),
            data_splitting=DataSplittingConfig(
                **data_splitting_dict,
                stratify=stratify_config
            ),
            preprocessing=PreprocessingConfig(
                **preprocessing_dict,
                normalization=normalization_config,
                resize=resize_config
            ),
            model=ModelConfig(
                **model_dict,
                dropout=dropout_config,
                batch_norm=batch_norm_config
            ),
            train=TrainConfig(
                **train_dict,
                early_stopping=early_stopping_config,
                reduce_lr=reduce_lr_config
            ),
            augmentation=AugmentationConfig(**config_dict['augmentation']),
            checkpoints=CheckpointsConfig(**config_dict['checkpoints']),
            metrics=MetricsConfig(**config_dict['metrics']),
            training_process=TrainingProcessConfig(
                **training_process_dict,
                cache=cache_config
            )
        )
    
    def update_from_args(self, args: Any) -> None:
        """
        Update config from command line arguments
        
        Args:
            args: Parsed command line arguments
        """
        if hasattr(args, 'batch_size') and args.batch_size:
            self.train.batch_size = args.batch_size
        
        if hasattr(args, 'image_size') and args.image_size:
            self.data.image_size = [args.image_size, args.image_size]
        
        if hasattr(args, 'epochs') and args.epochs:
            self.train.epochs = args.epochs
            
        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.train.learning_rate = args.learning_rate