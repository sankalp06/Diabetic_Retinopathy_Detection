import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

def load_csv(csv_file):
    return pd.read_csv(csv_file)

def fix_level_cat_format(row, separator=' ', replacement=', '):
    return eval(row.replace(separator, replacement))

def extract_labels(df, column_name='level_cat', fix_func=fix_level_cat_format):
    return np.array([fix_func(row) for row in df[column_name].values])

def extract_additional_features(df, feature_columns):
    return df[feature_columns].values

def normalize_features(features, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = scaler.transform(features)
    return scaled_features, scaler

def save_scaler(scaler, scaler_file):
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)

def load_and_preprocess_image(image_path, target_size=(224, 224), preprocessing_function=None):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    if preprocessing_function is not None:
        img = preprocessing_function(img)
    return img

def load_images(image_paths, preprocess_func):
    return np.array([preprocess_func(path) for path in image_paths])
