import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler

#timesplit, scale continous features, build model

df = pd.read_csv("LST-LSTM/YES_merged_with_labels.csv")

feats_cont = ['rain_1h', 'rain_3h', 'rain_6h', 'rain_12h', 'rain_24h', 'lst_z_filled']
feats_bin = ['lst_available']
feats = feats_cont + feats_bin
target = 'flood_label'
window = 48
horizon = 0
batch = 256
epochs = 50  


#timesplit
train_end = pd.Timestamp("2024-12-31 23:00:00")
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

Xtr, ytr = make_sequences(df_train, window, horizon)
Xva, yva = make_sequences(df_val, window, horizon)
Xte, yte = make_sequences(df_test, window, horizon)

def main():

    print("Shapes:", Xtr.shape, ytr.shape, Xva.shape, yva.shape, Xte.shape, yte.shape)
    
if __name__ == "__main__":
    main()