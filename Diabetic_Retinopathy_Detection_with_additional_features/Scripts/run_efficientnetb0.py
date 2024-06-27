import os
import pandas as pd
from utils.data_preprocessing import (
    load_csv, extract_labels, extract_additional_features, normalize_features, 
    save_scaler, load_images, split_data, load_and_preprocess_image
)
from models.efficientnetb0_model import create_efficientnetb0_model, efficientnetb0_preprocess_input
from utils.model_training import (
    create_additional_input, create_combined_model, 
    compile_and_train_model, evaluate_and_save_model
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

# Load and preprocess the dataset
csv_file = 'data/remaining_dataset.csv'
df = load_csv(csv_file)

# Extract labels and features
labels = extract_labels(df)
additional_features = extract_additional_features(df, [
    'HbA1c', 'Systolic_BP', 'Diastolic_BP', 'LDL', 'Duration', 
    'BMI', 'Glucose_SD', 'Triglycerides', 'Microalbuminuria', 
    'Smoking_years', 'Alcohol_frequency'
])

# Normalize the additional features
additional_features, scaler = normalize_features(additional_features)
scaler_file = 'models_saved/additional_features_scaler_efficientnetb0.pkl'
save_scaler(scaler, scaler_file)

# Load and preprocess images
images = load_images(df['full_path'], preprocess_func=lambda x: load_and_preprocess_image(
    x, target_size=(224, 224), preprocessing_function=efficientnetb0_preprocess_input))

# Split the data into training and testing sets
X_train_img, X_test_img, X_train_feat, X_test_feat, y_train, y_test = split_data(images, additional_features, labels)

# Create the EfficientNetB0 model
custom_layers_efficientnet = [GlobalAveragePooling2D(), Dense(256, activation='relu'), Dropout(0.5), Dense(128, activation='relu')]
efficientnet_input, efficientnet_output = create_efficientnetb0_model(custom_layers=custom_layers_efficientnet)

# Create the additional input layer for additional features
additional_input = create_additional_input((additional_features.shape[1],))

# Combine the EfficientNetB0 model with additional features
additional_layers = [Dense(128, activation='relu'), Dropout(0.5)]
model_efficientnet = create_combined_model(efficientnet_input, efficientnet_output, additional_input, additional_layers=additional_layers)

# Define custom callbacks
custom_callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Compile and train the model
history_efficientnet = compile_and_train_model(
    model_efficientnet, X_train_img, X_train_feat, y_train, X_test_img, X_test_feat, y_test, callbacks=custom_callbacks
)

# Evaluate and save the model, and save metrics to a text file
model_vgg16_save_path = 'models_saved/vgg16_with_additional_features.h5'
metrics_save_path = 'models_saved/vgg16_metrics.txt'
evaluate_and_save_model(model_efficientnet, X_test_img, X_test_feat, y_test, model_vgg16_save_path, metrics_save_path)
