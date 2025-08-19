import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from keras import layers, models, losses, metrics, optimizers
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from keras import backend as K #note this K is kinda deprecated 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob

#LOAD BATCH AND SEGMENTATION BATCHES
#load the patches 
patch_dir = "cnn_segmentation_patches"
input_dir  = os.path.join(patch_dir, "inputs")
label_dir = os.path.join(patch_dir, "labels")
metadata_dir = os.path.join(patch_dir, "patch_metadata.json")

patch_size = 64
num_channels = 16
batch_size = 16
num_classes = 2
epochs = 20

#loading function
def load_patch(input_filename, label_filename):
    X = np.load(os.path.join(input_dir, input_filename.decode())).astype(np.float32)
    y = np.load(os.path.join(label_dir, label_filename.decode()))
    y = np.where(np.isnan(y), 0, y).astype(np.uint8)
    y = np.clip(y, 0, 1)
    #print(f"Unique label values: {np.unique(y)}")
    return X, y

#create tf wrapper
def tf_wrapper(input_filename, label_filename):
    input_tensor, label_tensor = tf.numpy_function(
        load_patch, [input_filename, label_filename],
        [tf.float32, tf.uint8]
    )
    input_tensor.set_shape([patch_size, patch_size, num_channels])
    label_tensor.set_shape([patch_size, patch_size])
    label_tensor = tf.cast(label_tensor, tf.float32)  #ensure label is float for loss calculation
    return input_tensor, label_tensor

def dice_loss(y_true, y_pred, smooth=1e-6): #
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice_coeff = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice_coeff
    return loss

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1- y_pred)
    return tf.reduce_mean(-alpha * tf.pow(1 - pt, gamma) * tf.math.log(pt + K.epsilon()))

def total_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

class MeanIoUCustom(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight = None):
        y_pred = tf.cast(y_pred > 0.5, tf.int32)
        y_true =  tf.cast(y_true, tf.int32)
        return super().update_state(y_true, y_pred, sample_weight)

def unet_model(input_shape = (patch_size, patch_size, num_channels), num_classes = 2):
    inputs = tf.keras.Input(shape= input_shape)

    #encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    #decoder
    #note: upsampling v conv2Dtranspose > safer and easier to do upsampling... but tbc if underfitting or etc.
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    #output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def main():

    patch_dir = "cnn_segmentation_patches"
    input_dir  = os.path.join(patch_dir, "inputs")
    label_dir = os.path.join(patch_dir, "labels")
    metadata_dir = os.path.join(patch_dir, "patch_metadata.json")


    with open(metadata_dir, "r") as f:
        metadata = json.load(f)

    input_filenames = [item["input"] for item in metadata]
    label_filenames = [item["label"].replace("input", "label") for item in metadata]

    #testing to check imbalance dataset
    imbalance = sorted(glob.glob("cnn_segmentation_patches/labels/*.npy"))
    total_pixels = 0
    flood_pixels = 0

    for f in imbalance:
        label = np.load(f)
        total_pixels += label.size
        flood_pixels += np.sum(label == 1)  #1 is flood class

    print(f"flood pixels ratio: {flood_pixels / total_pixels:.6f}")

    '''

    def create_dataset(input_filenames, label_filenames, batch_size=32, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((input_filenames, label_filenames))
        dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=100) #buffer size = full shuffle for randomness
        
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        input_filenames, label_filenames, test_size=0.2, random_state=42)
    
    train_dataset = create_dataset(train_inputs, train_labels, batch_size=32)
    val_dataset = create_dataset(val_inputs, val_labels, batch_size=32, shuffle=False)
    
    model = unet_model(input_shape=(patch_size, patch_size, num_channels))

    model.compile(optimizer='adam', loss=total_loss, metrics=['accuracy', MeanIoUCustom(num_classes=num_classes)])
    
    callbacks = [
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    '''
    
    
    '''#shape batch shape
    for x_batch, y_batch in dataset.take(1):
        print(f"Input batch shape: {x_batch.shape}")
        print(f"Label batch shape: {y_batch.shape}")
 
    #test if the labels are valid
    def validate_labels(label_dir, label_filenames, num_classes=2):
        for filename in label_filenames:
            label = np.load(os.path.join(label_dir, filename))
            unique_vals = np.unique(label)
            if not np.all(np.isin(unique_vals, np.arange(num_classes))):
                print(f"Invalid labels in {filename}: {unique_vals}")

    validate_labels(label_dir, label_filenames, num_classes=num_classes)
    print("Total input patches:", len(input_filenames))
    print("Total input patches:", len(label_filenames))
    
    for i in range(3):
        x, y = dataset.take(1).as_numpy_iterator().next()
        print(f"x shape: {x.shape}, y shape: {y.shape}")
        print("Unique labels in y:", np.unique(y))

    '''
   

    #model.summary()    
    '''
    patch_folder = "cnn_patches"

    cnn_patches = sorted([
    os.path.join(patch_folder, f) 
    for f in os.listdir(patch_folder) 
    if f.endswith(".npy")
    ])

    '''
   # X = np.stack([np.load(f) for f in cnn_patches])

    # y = np.array([1] * 8 + [0] * 8)  # dummy binary labels


    # cnn_flood_model = build_cnn_model(output_type='classification')

   # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    #cnn_flood_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

   #below suposed to be after shape batch shape
    '''
    #seeing the patch shapes
    sample_inputs = sorted(os.listdir(input_dir))[:3]
    sample_labels = sorted(os.listdir(label_dir))[:3]

    for i, (inp_file, lbl_file) in enumerate(zip(sample_inputs, sample_labels)):
    inp_path = os.path.join(input_dir, inp_file)
    lbl_path = os.path.join(label_dir, lbl_file)

    inp = np.load(inp_path)
    lbl = np.load(lbl_path)

    print(f"sample {i+1}:")
    print(f"  Input shape: {inp.shape}")
    print(f"  Label shape: {lbl.shape}")
    print("-" * 30)

    '''


if __name__ == "__main__":
    main()



