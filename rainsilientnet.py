import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

#CNN to capture ....?

def build_cnn_model(input_shape=(64, 64, 4), output_type='classification'):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def main():
    patch_folder = "cnn_patches"

    cnn_patches = sorted([
    os.path.join(patch_folder, f) 
    for f in os.listdir(patch_folder) 
    if f.endswith(".npy")
])

    X = np.stack([np.load(f) for f in cnn_patches])

    y = np.array([1] * 8 + [0] * 8)  # dummy binary labels

    cnn_flood_model = build_cnn_model(output_type='classification')

    cnn_flood_model.fit(X, y, epochs=10)
    

if __name__ == "__main__":
    main()



