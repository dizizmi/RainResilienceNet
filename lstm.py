import numpy as np
import pandas as pd
import re

import sklearn
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras
from keras import layers, models

#constants
df = pd.read_csv("LST-LSTM/YES_merged_with_labels.csv")

feats_cont = ['rain_1h', 'rain_3h', 'rain_6h', 'rain_12h', 'rain_24h', 'lst_z_filled']
feats_bin = ['lst_available']
feats = feats_cont + feats_bin
target = 'flood_label'
window = 48
horizon = 0
batch = 256
epochs = 50  


def make_sequences(df_part, window=48, horizon=0):
    X, Y = [], []
    for sid, g in df_part.groupby("station_id"):
        g = g.sort_values("timestamp")
        A = g[feats].to_numpy(dtype=np.float32)
        L = g[target].to_numpy(dtype=np.float32)
        n = len(g)
        if n <= window + horizon: 
            continue
        for i in range(0, n - window - horizon):
            X.append(A[i:i+window])
            Y.append(L[i+window+horizon])
    if not X:
        return (np.empty((0,window,len(feats)), np.float32),
                np.empty((0,), np.float32))
    return np.stack(X), np.stack(Y)

#model
def ds(X,y,train=False,batch=256):
    d = tf.data.Dataset.from_tensor_slices((X,y))
    if train: d = d.shuffle(min(len(X),10000), reshuffle_each_iteration=True)
    return d.batch(batch).prefetch(tf.data.AUTOTUNE)


def build(win=48, dim=7, hidden=64, layers_num=2, dropout=0.2):
    inp = layers.Input((win,dim))
    x = inp
    for _ in range(layers_num-1):
        x = layers.LSTM(hidden, return_sequences=True)(x); x = layers.Dropout(dropout)(x)
    x = layers.LSTM(hidden)(x); x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp,out)


def main():


    df['timestamp'] = pd.to_datetime(df['timestamp'])

    #timesplit
    train_end = pd.Timestamp("2025-02-28 23:00:00")
    val_start = pd.Timestamp("2025-01-03 00:00:00")
    val_end   = pd.Timestamp("2025-03-31 23:00:00")
    test_start= pd.Timestamp("2025-04-03 00:00:00")


    df_train = df[df['timestamp'] < train_end].copy()
    df_val = df[(df['timestamp'] >= val_start) & (df['timestamp'] < val_end)].copy()
    df_test = df[df['timestamp'] >= test_start].copy()

    #[samples,time steps, features]
    #scale continuous features
    scaler = StandardScaler()
    df_train[feats_cont] = scaler.fit_transform(df_train[feats_cont])
    df_val[feats_cont] = scaler.transform(df_val[feats_cont])
    df_test[feats_cont] = scaler.transform(df_test[feats_cont])

    Xtr, ytr = make_sequences(df_train, window, horizon)
    Xva, yva = make_sequences(df_val, window, horizon)
    Xte, yte = make_sequences(df_test, window, horizon)

    print("Shapes:", Xtr.shape, ytr.shape, Xva.shape, yva.shape, Xte.shape, yte.shape)
    #Shapes: (551723, 48, 7) (551723,) (111783, 48, 7) (111783,) (200195, 48, 7) (200195,)
    

    train_ds = ds(Xtr,ytr,True,256)
    val_ds = ds(Xva,yva)
    test_ds = ds(Xte,yte)

    model = build(48,7,hidden=64,layers_num=2,dropout=0.2)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                metrics=[tf.keras.metrics.AUC(name="auc"),
                        tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                        tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    pos = float((ytr==1).sum()); neg = float((ytr==0).sum())
    class_weights = {0:1.0, 1:neg/(pos+1e-6)}

    cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=3, min_lr=1e-5),
    tf.keras.callbacks.ModelCheckpoint("best_lstm.keras", monitor="val_pr_auc", mode="max", save_best_only=True)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=50,
                        class_weight=class_weights, verbose=1)


if __name__ == "__main__":
    main()