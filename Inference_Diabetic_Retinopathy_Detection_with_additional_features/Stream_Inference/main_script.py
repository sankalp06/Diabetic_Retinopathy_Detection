import os
import numpy as np
import pickle
from data_utils import load_and_preprocess_image, load_scaler, fix_level_cat_format
from model_utils import load_model, predict_single_record

# Paths and files
model_path = r'E:\Computer_Vision_projects\inference_drd\Diabetic_Retinopathy_Detection\vgg16_with_additional_features.h5'
csv_file = r'E:\Computer_Vision_projects\inference_drd\Diabetic_Retinopathy_Detection\sampled_dataset.csv'
scaler_file = r'E:\Computer_Vision_projects\inference_drd\Diabetic_Retinopathy_Detection\additional_features_scaler.pkl'
left_eye_path = 'E:/Computer_Vision_projects/Diabetic_retinopathy_detection/train.zip.001/train/train/10_left.jpeg'
right_eye_path = 'E:/Computer_Vision_projects/Diabetic_retinopathy_detection/train.zip.001/train/train/10_right.jpeg'

# Additional features (example, replace with actual data)
additional_features = [
    10.896195065561075,  # HbA1c
    150.69693481299157,  # Systolic_BP
    97.58934460717728,   # Diastolic_BP
    193.71066317092425,  # LDL
    21.41599013207625,   # Duration
    41.20415312086392,   # BMI
    30.84676095056483,   # Glucose_SD
    311.23456504422853,  # Triglycerides
    120.01877115621907,  # Microalbuminuria
    23,                  # Smoking_years
    6,                   # Alcohol_frequency
]

# Load model and scaler
model = load_model(model_path)
scaler = load_scaler(scaler_file)

# Predict
predicted_classes, confidences = predict_single_record(model, left_eye_path, right_eye_path, additional_features, scaler)

# Example of using fix_level_cat_format from data_utils
# Load your dataframe
# df = pd.read_csv(csv_file)
# df['level_cat'] = df['level_cat'].apply(fix_level_cat_format)

# Output predictions
print(f'Predicted class for left eye: {predicted_classes[0]} with confidence {confidences[0]:.4f}')
print(f'Predicted class for right eye: {predicted_classes[1]} with confidence {confidences[1]:.4f}')
