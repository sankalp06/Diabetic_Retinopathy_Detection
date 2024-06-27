from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout

def create_vgg16_model(input_shape=(224, 224, 3), base_model_weights='imagenet', include_top=False, custom_layers=None):
    base_model = VGG16(weights=base_model_weights, include_top=include_top, input_shape=input_shape)
    x = base_model.output
    if custom_layers is not None:
        for layer in custom_layers:
            x = layer(x)
    else:
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
    return base_model.input, x

def vgg16_preprocess_input(x):
    from tensorflow.keras.applications.vgg16 import preprocess_input
    return preprocess_input(x)