
import os
import warnings
import numpy as np

# Suppress TensorFlow logs and oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler

# Suppress all warnings
warnings.filterwarnings('ignore')

# Further suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

# Load your saved model
model = tf.keras.models.load_model('E:/Computer_Vision_projects/vgg16_with_additional_features_multiclass.h5')

# User input for additional features and image paths
left_eye_path = 'E:/Computer_Vision_projects/Diabetic_retinopathy_detection/train.zip.001/train/train/10_left.jpeg'
right_eye_path = 'E:/Computer_Vision_projects/Diabetic_retinopathy_detection/train.zip.001/train/train/10_right.jpeg'


# Function to load and preprocess image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

# Function to predict class for a single record
def predict_single_record(left_eye_path, right_eye_path, additional_features):
    # Load and preprocess images
    left_eye_img = load_and_preprocess_image(left_eye_path)
    right_eye_img = load_and_preprocess_image(right_eye_path)

    # Convert additional features to numpy array and scale
    additional_features = np.array(additional_features).reshape(1, -1)
    scaler = StandardScaler()
    additional_features_scaled = scaler.fit_transform(additional_features)

    # Repeat additional features to match the number of images (left and right eye)
    additional_features_repeated = np.repeat(additional_features_scaled, 2, axis=0)

    # Stack images to create a batch of size 2 (left and right eye)
    images = np.stack([left_eye_img, right_eye_img], axis=0)

    # Temporarily suppress warnings during prediction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Predict for both eyes
        predictions = model.predict([images, additional_features_repeated])
    
    # Assuming predictions are probabilities for each class, extract the predicted class and confidence
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    return predicted_classes, confidences


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
    154.44188252476644   # BP
]

# Get predictions and confidences
predicted_classes, confidences = predict_single_record(left_eye_path, right_eye_path, additional_features)

# Output predictions and confidences for left and right eye
print(f'Predicted class for left eye: {predicted_classes[0]} with confidence {confidences[0]:.4f}')
print(f'Predicted class for right eye: {predicted_classes[1]} with confidence {confidences[1]:.4f}')
