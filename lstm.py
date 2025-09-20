import numpy as np
import pandas as pd
import re

import sklearn
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, LayerNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, classification_report


SEQ_LEN = 48
FEATURES = ["rainfall_mm","rain_1h","rain_3h","rain_6h","rain_12h","rain_24h"]


df = pd.read_csv("LST-LSTM/ultra_flood_rainfall.csv", parse_dates=["timestamp"])
df = df.sort_values(["station_id","timestamp"]).reset_index(drop=True)


train_end   = pd.Timestamp("2024-12-31 23:00:00")
val_start   = pd.Timestamp("2025-01-03 00:00:00")
val_end     = pd.Timestamp("2025-03-31 23:00:00")
test_start  = pd.Timestamp("2025-04-03 00:00:00")

df_tr = df[df["timestamp"] <= train_end].copy()
df_va = df[(df["timestamp"] >= val_start) & (df["timestamp"] <= val_end)].copy()
df_te = df[df["timestamp"] >= test_start].copy()


scaler = StandardScaler()
df_tr[FEATURES] = scaler.fit_transform(df_tr[FEATURES])
df_va[FEATURES] = scaler.transform(df_va[FEATURES])
df_te[FEATURES] = scaler.transform(df_te[FEATURES])

#causal sequence
def make_sequences(d, seq_len=SEQ_LEN, fcols=FEATURES, ycol="flood_label"):
    X, y = [], []
    for sid, g in d.groupby("station_id"):
        A = g[fcols].to_numpy(np.float32)
        L = g[ycol].to_numpy(np.float32)
        n = len(g)
        if n <= seq_len: 
            continue
        for i in range(n - seq_len):
            X.append(A[i:i+seq_len])
            y.append(L[i+seq_len-1])
    if not X:
        return np.empty((0,seq_len,len(fcols)), np.float32), np.empty((0,), np.float32)
    return np.stack(X), np.stack(y)

Xtr, ytr = make_sequences(df_tr)
Xva, yva = make_sequences(df_va)
Xte, yte = make_sequences(df_te)


model = Sequential([
    LayerNormalization(axis=-1),
    LSTM(64, input_shape=(SEQ_LEN, len(FEATURES))),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4, clipnorm=1.0),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
             tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)


def ds(X,y,train=False,batch=256):
    d = tf.data.Dataset.from_tensor_slices((X,y))
    if train: d = d.shuffle(min(len(X),10000), reshuffle_each_iteration=True)
    return d.batch(batch).prefetch(tf.data.AUTOTUNE)


   
def main():
    train_ds = ds(Xtr,ytr,train=True)
    val_ds   = ds(Xva,yva)
    test_ds  = ds(Xte,yte)

    pos = float((ytr==1).sum()); neg = float((ytr==0).sum())
    w_pos = min(50.0, neg/(pos+1e-6))   #cap 
    class_weights = {0:1.0, 1:w_pos}

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=3, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint("best_lstm.keras", monitor="val_pr_auc", mode="max", save_best_only=True)
    ]

    #train
    history = model.fit(train_ds, validation_data=val_ds, epochs=30,
                        class_weight=class_weights, callbacks=cbs, verbose=1)

    #threshold on tuning
    y_val_prob = model.predict(val_ds).ravel()
    prec, rec, thr = precision_recall_curve(yva, y_val_prob)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    best_idx = np.nanargmax(f1)
    best_thr = float(thr[best_idx])
    print(f"Best F1 threshold={best_thr:.3f}  P={prec[best_idx]:.3f} R={rec[best_idx]:.3f}")

    #evaluate
    y_test_prob = model.predict(test_ds).ravel()
    y_test_pred = (y_test_prob >= best_thr).astype(int)

    print(classification_report(yte, y_test_pred, digits=3))

if __name__ == "__main__":
    main()