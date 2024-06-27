import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

def load_scaler(scaler_file):
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def fix_level_cat_format(row):
    return eval(row.replace(' ', ', '))
