# Diabetic Retinopathy Detection with Additional Features

This project focuses on leveraging deep learning models for Diabetic Retinopathy detection using retinal images and additional patient features. The integration of additional features alongside image data enhances the model's ability to predict disease severity accurately.

## Overview

This repository provides a robust framework for developing, training, and evaluating deep learning models tailored for Diabetic Retinopathy detection. Key features include:

- Integration of pre-trained models like VGG16, ResNet, Inception, and EfficientNetB0 for feature extraction from retinal images.
- Incorporation of additional patient-specific features such as HbA1c, blood pressure, lipid profile, and lifestyle factors.
- Modularized scripts for data preprocessing, model building, training, evaluation, and prediction.
- Storage and management of trained models, scaler objects, and evaluation metrics.

## Additional Features Integration

In the context of medical imaging and diagnostics, incorporating additional patient features beyond image data is crucial. These features provide contextual information about the patient's health status, which complements the visual data extracted from retinal images. Here’s how these additional features contribute:

- **HbA1c (Glycated Hemoglobin):** Reflects the average blood sugar level over the past 2-3 months, offering insights into long-term diabetes management.
- **Blood Pressure (Systolic and Diastolic):** Indicators of cardiovascular health and potential complications related to hypertension.
- **LDL (Low-Density Lipoprotein):** A primary marker for assessing cholesterol levels and cardiovascular risk.
- **Duration of Diabetes:** Provides information on disease progression and its impact on retinal health.
- **BMI (Body Mass Index):** Measures body fat based on height and weight, influencing diabetes management and associated risks.
- **Glucose Standard Deviation:** Indicates glycemic variability, offering insights beyond average glucose levels.
- **Triglycerides:** Another lipid marker associated with cardiovascular risk factors.
- **Microalbuminuria:** Predictive of diabetic nephropathy, a common complication impacting retinal health.
- **Smoking and Alcohol Habits:** Lifestyle factors influencing overall health and diabetes management.

These features collectively enrich the model’s ability to make informed predictions about Diabetic Retinopathy severity, enabling personalized patient care and intervention strategies.

## Folder Structure for training model

```
.
├── README.md                       # This README file
├── train_scripts/                  # Scripts for training models
│   ├── train_vgg16.py              # Script for training VGG16 model
│   ├── train_resnet.py             # Script for training ResNet model
│   ├── train_inception.py          # Script for training Inception model
│   ├── train_efficientnet.py       # Script for training EfficientNet model
│   └── ...
├── utils/                          # Utility scripts and modules
│   ├── data_utils.py               # Functions for data loading and preprocessing
│   ├── model_utils.py              # Functions for building and compiling models
│   ├── evaluation_utils.py         # Functions for model evaluation and metrics
│   ├── predict_utils.py            # Functions for making predictions with trained models
│   ├── scaler_utils.py             # Functions for handling scalers for additional features
│   └── ...
├── models/                         # Saved models and scalers
│   ├── vgg16_with_additional_features.h5    # Saved VGG16 model with additional features
│   ├── resnet_with_additional_features.h5   # Saved ResNet model with additional features
│   ├── inception_with_additional_features.h5    # Saved Inception model with additional features
│   ├── efficientnet_with_additional_features.h5 # Saved EfficientNet model with additional features
│   ├── additional_features_scaler.pkl   # Scaler object for additional features
│   └── ...
├── data/                           # Sample dataset and CSV files
│   └── train_dataset.csv         # Sample CSV dataset with patient data
├── metrics/                        # Performance metrics and evaluation results
│   ├── vgg16_metrics.txt           # Metrics for VGG16 model
│   ├── resnet_metrics.txt          # Metrics for ResNet model
│   ├── inception_metrics.txt       # Metrics for Inception model
│   ├── efficientnet_metrics.txt    # Metrics for EfficientNet model
│   ├── classification_reports/     # Detailed classification reports
│   │   ├── vgg16_classification_report.txt
│   │   ├── resnet_classification_report.txt
│   │   ├── inception_classification_report.txt
│   │   ├── efficientnet_classification_report.txt
│   │   └── ...
│   └── confusion_matrices/         # Confusion matrices
│       ├── vgg16_confusion_matrix.txt
│       ├── resnet_confusion_matrix.txt
│       ├── inception_confusion_matrix.txt
│       ├── efficientnet_confusion_matrix.txt
│       └── ...
├── requirements.txt                # List of Python dependencies
└── LICENSE                         # License information

```

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   - Ensure TensorFlow, Keras, scikit-learn, NumPy, and Pandas are installed along with other required dependencies.

## Usage

### Training the Models

1. Navigate to `train_scripts/`.
2. Run the desired training script for each model:
   ```bash
   python train_vgg16.py
   python train_resnet.py
   python train_inception.py
   python train_efficientnet.py
   ```
   - Modify script parameters (epochs, batch size, etc.) as needed.

### Predicting with Trained Models

1. Update paths and patient data in `predict_utils.py`.
2. Run the prediction script:
   ```bash
   python predict_utils.py
   ```

### Evaluating Model Performance

1. After training, metrics such as accuracy, precision, recall, and F1-score are saved in `metrics/`.
2. Detailed classification reports and confusion matrices are also available in their respective directories.
