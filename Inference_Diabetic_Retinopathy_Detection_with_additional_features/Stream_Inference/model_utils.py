import os
import warnings
import numpy as np
import tensorflow as tf

# Suppress TensorFlow logs and oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Suppress all warnings
warnings.filterwarnings('ignore')

# Further suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects=None, compile=True, safe_mode=True)
    return model

def predict_single_record(model, left_eye_path, right_eye_path, additional_features, scaler):
    left_eye_img = load_and_preprocess_image(left_eye_path)
    right_eye_img = load_and_preprocess_image(right_eye_path)

    additional_features = np.array(additional_features).reshape(1, -1)
    additional_features_scaled = scaler.transform(additional_features)
    additional_features_repeated = np.repeat(additional_features_scaled, 2, axis=0)

    images = np.stack([left_eye_img, right_eye_img], axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = model.predict([images, additional_features_repeated])

    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    return predicted_classes, confidences
