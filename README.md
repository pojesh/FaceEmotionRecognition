# FaceEmotionRecognition

A deep learning project for facial emotion recognition using the AffectNet dataset and custom deep convolutional neural networks (DCNNs). The repository includes data preprocessing, augmentation, model training, evaluation, and result visualization.

## Project Structure

- `affectnet/YOLO_format/`  
  Contains the AffectNet dataset in YOLO format, including images and label files for training, validation, and testing.  
  - `train/`, `valid/`, `test/`: Original dataset splits  
  - `train_augmented/`: Augmented training data for class balancing  
  - `class_distribution.png`, `data.yaml`: Dataset statistics and configuration

- `models/`  
  - `dcnn_architecture.py`: DCNN model architecture definition  
  - `dcnn_run.ipynb`: Jupyter notebook for training, validating, and testing the DCNN model  
  - `deepemotion_model_architecture.py`, `deepemotion_run.ipynb`: Additional model architectures and runs

- `outputs/`  
  - `dcnn_result/`: Results from DCNN experiments (trained models, result images, metrics, confusion matrix, etc.)  
  - `de_epoch10_acc53/`, `de_epoch5_acc50/`: Results from other model runs

- `dcnn_docs/`  
  - `dcnn_report.pdf`, `dcnn_summary.txt`: Project report and summary

- `train_data_process.ipynb`: Data preprocessing and augmentation notebook  
- `data_distribution.ipynb`: Data analysis and visualization  
- `dataset_description.txt`: Dataset details

## Features

- **Data Augmentation**: Balances class distribution using flipping, brightness/contrast adjustment, and rotation.
- **Custom DCNN Model**: Built with TensorFlow/Keras for multi-class emotion classification.
- **Evaluation**: Includes accuracy, loss, classification report, and confusion matrix.
- **Visualization**: Training/validation curves and confusion matrices are saved as images.

## Usage

### 1. Data Preparation

Ensure the AffectNet dataset is available in the `affectnet/YOLO_format/` directory, organized as required.

### 2. Data Augmentation

Run `train_data_process.ipynb` to augment the training data and balance class distribution.

### 3. Model Training

Use `models/dcnn_run.ipynb` to train the DCNN model.  
- Configuration (paths, hyperparameters) can be set at the top of the notebook.
- The model is saved as `outputs/dcnn_result/dcnn_model1.h5.keras`.

### 4. Evaluation

After training, evaluation metrics and confusion matrices are generated and saved in the `outputs/dcnn_result/` directory.  
- Example results (from `results.txt`):  
  - Test Accuracy: ~66%  
  - Per-class precision, recall, F1-score  
  - Confusion matrix

### 5. Documentation

See `dcnn_docs/dcnn_report.pdf` for a detailed project report, including methodology, experiments, and analysis.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- scikit-learn
- Pillow
- OpenCV
- Matplotlib
- Seaborn

Install dependencies with:

```bash
pip install tensorflow keras numpy scikit-learn pillow opencv-python matplotlib seaborn