import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from data_utils import load_csv, load_and_preprocess_image, load_scaler, fix_level_cat_format
from model_utils import predict_records, calculate_metrics

# Suppress TensorFlow logs and oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Load your saved model
model_path = r'E:\Computer_Vision_projects\inference_drd\Diabetic_Retinopathy_Detection\vgg16_with_additional_features.h5'
model = tf.keras.models.load_model(model_path)

# Load the CSV file
csv_file = r'E:\Computer_Vision_projects\inference_drd\Diabetic_Retinopathy_Detection\sampled_dataset.csv'
df = load_csv(csv_file)

# Path to the scaler file
scaler_file = r'E:\Computer_Vision_projects\inference_drd\Diabetic_Retinopathy_Detection\additional_features_scaler.pkl'

# Load the scaler
scaler = load_scaler(scaler_file)

# Predict for all records in the CSV
true_classes, predicted_classes = predict_records(model, df, scaler)

# Calculate metrics
accuracy, precision, recall, f1, conf_matrix, class_report = calculate_metrics(true_classes, predicted_classes)

# Print and save metrics to a text file
metrics_save_path = r'E:\Computer_Vision_projects\inference_drd\Diabetic_Retinopathy_Detection\metrics.txt'

with open(metrics_save_path, 'w') as f:
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write(f'Precision: {precision:.4f}\n')
    f.write(f'Recall: {recall:.4f}\n')
    f.write(f'F1 Score: {f1:.4f}\n\n')
    f.write('Confusion Matrix:\n')
    f.write(f'{conf_matrix}\n\n')
    f.write('Classification Report:\n')
    f.write(class_report)

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Confusion matrix
print('Confusion Matrix:')
print(conf_matrix)

# Classification report
print('Classification Report:')
print(class_report)
