# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 10:28:00 2025
@author: Girraj
"""

# Section 1: Imports and Seeding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import time
import random
import matplotlib
import matplotlib.dates as mdates
matplotlib.rcParams.update({'font.size': 1, 'font.family': 'Times New Roman'})

M_Q = 0  # Set to 1 if using multi-quantile model
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# Section 2: Load and Prepare Data
GenData = pd.read_csv("CaISO_GenData_2024_15min.csv")
GenData['dateCol'] = pd.to_datetime(GenData['Date'], dayfirst=True)

samples = 24 * 4  # 1 day = 96 samples (15-min intervals)
Test_H = 7 * samples  # 7 days = 672 samples

Output = 'Wind+Curtail'

WS = np.array(GenData['WindSpeed'])

WS_lag1 = np.zeros(len(WS))
WS_lag2 = np.zeros(len(WS))
WS_lag3 = np.zeros(len(WS))
WS_lag4 = np.zeros(len(WS))
WS_lag5 = np.zeros(len(WS))
WS_lag6 = np.zeros(len(WS))

L = len(WS)

WS_lag1[0:L-1] = WS[1:]
WS_lag2[0:L-2] = WS[2:]
WS_lag3[0:L-3] = WS[3:]
WS_lag4[0:L-4] = WS[4:]
WS_lag5[0:L-5] = WS[5:]
WS_lag6[0:L-6] = WS[6:]

GenData['W2'] = np.square(WS)
GenData['W3'] = np.square(WS) * WS
GenData['WS_lag1'] = WS_lag1
GenData['WS_lag2'] = WS_lag2
GenData['WS_lag3'] = WS_lag3
GenData['WS_lag4'] = WS_lag4
GenData['WS_lag5'] = WS_lag5
GenData['WS_lag6'] = WS_lag6
Traindate = "2024-06-30 23:45:00"
trInd = np.where(GenData['dateCol'] == Traindate)[0].item()

testInd = trInd + Test_H

Inputs = ['Temperature', 'WindDirection', 'WindSpeed'] #, 'WS_lag1', 'WS_lag2', 'WS_lag3', 'WS_lag4', 'WS_lag5', 'WS_lag6']
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(GenData[Inputs])
y = scaler_y.fit_transform(GenData[[Output]])

# Section 3: Sequences Preparation
seq_len = 6

def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X, y, seq_len)
X_train = X_seq[:trInd - seq_len]
y_train = y_seq[:trInd - seq_len]
X_test = X_seq[trInd - seq_len:testInd - seq_len]
y_test = y_seq[trInd - seq_len:testInd - seq_len]
print("Data preparation successful")

# Section 4: Quantile LSTM Model
if M_Q:
    class MultiQuantileLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=128*8, num_layers=2, quantiles=[0.05, 0.5, 0.95]):
            super().__init__()
            self.quantiles = quantiles
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, len(quantiles))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.linear(out[:, -1, :])

    def multi_quantile_loss(y_pred, y_true, quantiles=[0.05, 0.5, 0.95]):
        loss = 0
        for i, q in enumerate(quantiles):
            error = y_true - y_pred[:, i:i+1]
            loss += torch.mean(torch.max(q * error, (q - 1) * error))
        return loss

else:
    class QuantileLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=128*8, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.linear(out[:, -1, :])

    def quantile_loss(pred, true, q):
        error = true - pred
        return torch.mean(torch.max(q * error, (q - 1) * error))

# Section 5: Train Model
if M_Q:
    def train_model(X_train, y_train, epochs=100, patience=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiQuantileLSTM(input_size=X_train.shape[2]).to(device)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_loss = float('inf')
        wait = 0
        start = time.time()

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for xb, yb in loader:
                preds = model(xb)
                loss = multi_quantile_loss(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            print(f"Epoch {epoch+1} | Loss={epoch_loss:.5f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered")
                    break

        model.load_state_dict(best_model)
        total_time = time.time() - start
        print(f"Training completed in {total_time:.2f} seconds")
        return model.cpu(), total_time

else:
    def train_model(X_train, y_train, q, epochs=100, patience=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = QuantileLSTM(input_size=X_train.shape[2]).to(device)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_loss = float('inf')
        wait = 0
        start = time.time()

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for xb, yb in loader:
                preds = model(xb)
                loss = quantile_loss(preds, yb, q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            print(f"Quantile {q} | Epoch {epoch+1} | Loss={epoch_loss:.5f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered")
                    break

        model.load_state_dict(best_model)
        total_time = time.time() - start
        print(f"Training for quantile {q} completed in {total_time:.2f} seconds")
        return model.cpu(), total_time

# Section 6: Train and Predict
if M_Q:
    model, t_all = train_model(X_train, y_train)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test_tensor).numpy()

    q05 = scaler_y.inverse_transform(preds[:, 0:1])
    q50 = scaler_y.inverse_transform(preds[:, 1:2])
    q95 = scaler_y.inverse_transform(preds[:, 2:3])
    y_true = scaler_y.inverse_transform(y_test)

    print(f"Total Training Time (Multi-Quantile): {t_all:.2f} seconds")

else:
    model_q05, t05 = train_model(X_train, y_train, 0.05)
    model_q50, t50 = train_model(X_train, y_train, 0.5)
    model_q95, t95 = train_model(X_train, y_train, 0.95)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        q05 = model_q05(X_test_tensor).numpy()
        q50 = model_q50(X_test_tensor).numpy()
        q95 = model_q95(X_test_tensor).numpy()

    q05 = scaler_y.inverse_transform(q05)
    q50 = scaler_y.inverse_transform(q50)
    q95 = scaler_y.inverse_transform(q95)
    y_true = scaler_y.inverse_transform(y_test)

    total_time_all = t05 + t50 + t95
    print(f"Total Training Time (All Quantiles): {total_time_all:.2f} seconds")


# Section 7: Plot Results
plot_dates = GenData['dateCol'].iloc[trInd + 1:testInd + 1].reset_index(drop=True)
mask = (plot_dates >= pd.to_datetime("2024-07-02")) & (plot_dates < pd.to_datetime("2024-07-04"))

plot_dates = plot_dates[mask]
q05 = q05[mask.values]
q50 = q50[mask.values]
q95 = q95[mask.values]
y_true = y_true[mask.values]

plt.figure(figsize=(12, 6))
plt.plot(plot_dates, y_true, label='Actuals', linewidth=2, color='blue', antialiased=False)
plt.plot(plot_dates, q05, label='Q05', color='orange', antialiased=False)
plt.plot(plot_dates, q50, label='Q50', color='green', antialiased=False)
plt.plot(plot_dates, q95, label='Q95', color='red', antialiased=False)
plt.title("Quantile LSTM")
plt.xlabel("Time (15 minutes)")
plt.ylabel("Wind Generation (MW)")
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
plt.grid(True)
plt.tight_layout()
plt.show()


alg="QLSTM"
Horz= " 24H"
td=pd.to_datetime(plot_dates,dayfirst=True)
plt.figure("RF_Q")
plt.plot(y_true)
plt.plot(td,q05)
plt.plot(td,q50)
plt.plot(td,q95)
plt.legend(['Actuals','Q5','Q50','Q95'], fontsize=12, loc='upper center', ncol=4)
plt.xlabel('Time (15 minutes)',  fontsize=15, fontweight='bold')
plt.ylabel('Wind Generation (MW)',  fontsize=15, fontweight='bold')
plt.title(alg+Horz,  fontsize=15, fontweight='bold')
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
plt.yticks(fontsize=13)
plt.xticks(rotation=40, ha='right', fontsize=13)
plt.tight_layout()
plt.show()

# Section 8: Metrics
def pinball(y, q, tau):
    return np.mean(np.maximum(tau * (y - q), (tau - 1) * (y - q)))

loss_q05 = pinball(y_true, q05, 0.05)
loss_q50 = pinball(y_true, q50, 0.5)
loss_q95 = pinball(y_true, q95, 0.95)
coverage = np.mean((y_true >= q05) & (y_true <= q95))
sharpness = np.mean(q95 - q05)

print(f"Q05 Loss: {loss_q05:.4f} | Q50 Loss: {loss_q50:.4f} | Q95 Loss: {loss_q95:.4f}")
print(f"Coverage: {coverage:.2f} | Sharpness: {sharpness:.2f}")
print("Evaluation complete")