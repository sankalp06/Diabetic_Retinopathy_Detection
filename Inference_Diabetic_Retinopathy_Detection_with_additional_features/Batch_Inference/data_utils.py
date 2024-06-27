import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Function to load CSV data
def load_csv(csv_file):
    return pd.read_csv(csv_file)

# Function to load and preprocess image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

# Function to load scaler
def load_scaler(scaler_file):
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# Function to fix the format of level_cat and extract labels
def fix_level_cat_format(row):
    return eval(row.replace(' ', ', '))
