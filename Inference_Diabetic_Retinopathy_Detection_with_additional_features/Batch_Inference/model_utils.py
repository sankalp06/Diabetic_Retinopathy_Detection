import numpy as np
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from data_utils import load_and_preprocess_image, fix_level_cat_format
# Suppress warnings and TensorFlow logs
warnings.filterwarnings('ignore')

# Function to predict using the model
def predict_records(model, df, scaler):
    images = []
    additional_features = []
    true_labels = []

    for index, row in df.iterrows():
        left_eye_img = load_and_preprocess_image(row['full_path'])
        images.append(left_eye_img)
        additional_features.append([
            row['HbA1c'], row['Systolic_BP'], row['Diastolic_BP'], row['LDL'], 
            row['Duration'], row['BMI'], row['Glucose_SD'], row['Triglycerides'], 
            row['Microalbuminuria'], row['Smoking_years'], row['Alcohol_frequency']
        ])
        true_labels.append(fix_level_cat_format(row['level_cat']))  # Fix the format of level_cat

    # Convert to numpy arrays
    images = np.array(images)
    additional_features = np.array(additional_features)
    true_labels = np.array(true_labels)

    # Scale additional features using the loaded scaler
    additional_features_scaled = scaler.transform(additional_features)

    # Predict using the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = model.predict([images, additional_features_scaled])

    # Extract the predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)

    return true_classes, predicted_classes

# Function to calculate evaluation metrics
def calculate_metrics(true_classes, predicted_classes):
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted')
    recall = recall_score(true_classes, predicted_classes, average='weighted')
    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    class_report = classification_report(true_classes, predicted_classes, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'])

    return accuracy, precision, recall, f1, conf_matrix, class_report
