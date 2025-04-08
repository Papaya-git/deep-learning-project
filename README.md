# Binary Image Classification for Tounum

A streamlined project for binary image classification using Convolutional Neural Networks (CNN) with TensorFlow. This project distinguishes between photos and non-photos (paintings, schematics, sketches, text) for Tounum's document digitization workflow.

## Project Structure

- **configs/**: Configuration files for model and training parameters
  - `config.yaml`: Main configuration file with data, model, and training settings

- **src/**: Source code
  - **data/**: Data handling
    - `data_loader.py`: Loads and splits image data
    - `augmentation.py`: Implements various data augmentation strategies
    - `create_dataset.py`: Creates datasets with specified augmentation
  - **models/**: Model architecture
    - `cnn.py`: Defines the CNN architecture for binary classification
  - **train/**: Training logic
    - `trainer.py`: Implements the training loop and model compilation
  - **utils/**: Utility functions
    - `config.py`: Configuration management using dataclasses
    - `callbacks.py`: Training callbacks for checkpointing, early stopping, etc.
  - **evaluation/**: Evaluation tools
    - `metrics.py`: Comprehensive metrics for binary classification
  - `main.py`: Entry point that ties everything together

- **notebooks/**: Jupyter notebooks
  - `binary_classification_demo.ipynb`: Demo notebook for presentation

- **requirements.txt**: Python dependencies

## How It Works

1. **Configuration**: All parameters are defined in `configs/config.yaml` and loaded using the `Config` class in `utils/config.py`.

2. **Data Loading**: The system loads data from a single dataset directory containing 5 original classes (photo, paintings, schematics, sketch, text) and splits it into training, validation, and test sets. It converts the multi-class data into a binary classification problem (photo vs. non-photo).

3. **Model Definition**: The `build_model` function creates a CNN architecture for binary classification.

4. **Training**: The `Trainer` class handles model compilation and training with appropriate callbacks.

5. **Evaluation**: The `evaluate_model` function generates comprehensive metrics for binary classification.

## Training & Evaluation Workflow

### 1. Preparing your data

**Prepare your data**:
   Organize your images in the following structure:
   ```
   data/
   └── dataset/
       ├── photo/
       ├── paintings/
       ├── schematics/
       ├── sketch/
       └── text/
   ```

### 1. Training the Model

**Update configuration**:
   Edit `configs/config.yaml` to match your dataset and requirements.

**Using Docker (Recommended):**
```bash
# Start training with GPU support
docker compose up trainer

# For CPU-only training:
docker compose up trainer --force-recreate --no-deps
```

**Native Python:**
```bash
python src/main.py --mode train --config configs/config.yaml
```

Training will automatically:
- Save checkpoints to `checkpoints/` directory
- Apply early stopping if validation accuracy plateaus
- Log metrics in TensorBoard format

### 2. Evaluating Model Performance

**Check training metrics:**
```bash
tensorboard --logdir logs/
```

**Run test evaluation:**
```bash
# Using Docker
docker compose up predict

# Native Python
python src/main.py --mode test --model_path checkpoints/model_latest
```

Evaluation outputs:
- Classification report (precision/recall/F1)
- ROC curve and AUC score
- Confusion matrix
- Accuracy metrics on test set

**Visualization of results**:
- Automatic plot generation in `reports/figures/` containing:
  - `confusion_matrix_*.png`: Heatmap of true vs predicted classifications
  - `roc_curve_*.png`: ROC curve with AUC score
  - `precision_recall_*.png`: Precision-Recall curve
  
  Example file structure:
  ```bash
  reports/figures/
  ├── confusion_matrix_20231122_1430.png
  ├── roc_curve_20231122_1430.png
  └── precision_recall_20231122_1430.png
  ```

### 3. Making Predictions

The Docker Compose file is preconfigured for predictions:
```bash
# Ensure model path is correct in docker-compose.yaml
docker-compose up predict
```

For custom predictions:
```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints tounum-cnn \
  python src/main.py --mode test --model_path checkpoints/model_YYYYMMDD_HHMMSS
```

## Docker Management

- Build fresh containers: `docker compose build`
- View training logs: `docker compose logs -f trainer`
- Clean up resources: `docker compose down --volumes --rmi all`
- Monitor GPU usage: `nvidia-smi`

## Customization

- **Model Architecture**: Modify `src/models/cnn.py` to change the CNN architecture.
- **Data Augmentation**: Adjust the augmentation parameters in `src/data/augmentation.py`.
- **Training Parameters**: Update `configs/config.yaml` to change learning rate, batch size, etc.
- **Dataset Split**: Adjust the train/validation/test split ratios in `configs/config.yaml`.
