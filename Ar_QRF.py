# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 10:16:25 2025

@author: Girraj
"""

# import torch
# print(torch.__version__)

import pandas as pd
import matplotlib
import matplotlib.dates as mdates
import numpy as np
from sklearn.neural_network import MLPRegressor
import datetime as dt
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.stattools import pacf 
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats.stats import pearsonr   
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression, LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
matplotlib.rcParams.update({'font.size': 1, 'font.family': 'Times New Roman'})
import os

plt.rcParams.update({'font.size': 18})  # Set font size globally


alg = 'ARQRF ' ## replace with AR_QRF
Horz = '12H' ## '6H' for six hours ahead'

samples =12*4 ## number of samples for day ahead (forecast horizon)

Test_H = 7*samples ## 7  moving window forecast

ar_feat=1 ## 0 for QRF and 1 for AR_QRF

GenData = pd.read_csv("D:/movies/CaISO_GenData_2024_15min.csv")

# Wdata = pd.read_csv("C:/Users/navee/Downloads/5Loc_Weather_2024_15min.csv")

Output = 'Wind+Curtail'



def compute_sharpness(lower_bounds, upper_bounds):
    """
    Compute sharpness (average interval width) for prediction intervals.

    Parameters:
    - lower_bounds: np.array of shape (N,) — Lower quantile predictions (e.g., 5th percentile)
    - upper_bounds: np.array of shape (N,) — Upper quantile predictions (e.g., 95th percentile)

    Returns:
    - sharpness: float — Mean width of the prediction intervals
    """
    if len(lower_bounds) != len(upper_bounds):
        raise ValueError("Lower and upper bounds must have the same length.")
        
    interval_widths = upper_bounds - lower_bounds
    sharpness = np.mean(interval_widths)
    return sharpness

WS = np.array(GenData['WindSpeed'])

WS_lag1=np.zeros(len(WS))
WS_lag2=np.zeros(len(WS))
WS_lag3=np.zeros(len(WS))
WS_lag4=np.zeros(len(WS))
WS_lag5=np.zeros(len(WS))
WS_lag6=np.zeros(len(WS))

L=len(WS)

WS_lag1[0:L-1]=WS[1:]
WS_lag2[0:L-2]=WS[2:]
WS_lag3[0:L-3]=WS[3:]
WS_lag4[0:L-4]=WS[4:]
WS_lag5[0:L-5]=WS[5:]
WS_lag6[0:L-6]=WS[6:]



GenData['W2'] =np.square(WS)
GenData['W3'] =np.square(WS)*WS
GenData['WS_lag1']=WS_lag1
GenData['WS_lag2']=WS_lag2
GenData['WS_lag3']=WS_lag3
GenData['WS_lag4']=WS_lag4
GenData['WS_lag5']=WS_lag5
GenData['WS_lag6']=WS_lag6

dateCol=pd.to_datetime(GenData['Date'],dayfirst=True)
GenData['dateCol'] = dateCol

      
        
Traindate = "2024-06-30 23:45:00"



ForecastHorizon=samples  ## day-Ahead

Inputs = ['Temperature',
'WindDirection', 'WindSpeed','WS_lag1', 'WS_lag2', 'WS_lag3', 'WS_lag4', 'WS_lag5', 'WS_lag6']

  ## can be replaced with any variable of interest



## 'Wind', 'Net Load',
     #  'Renewables', 'Nuclear', 'Large Hydro', 'Imports', 'Generation'


trInd=int(np.where(GenData['dateCol']==Traindate)[0])

testind = trInd+Test_H

X= GenData[Inputs]



X_tr =X.iloc[0:trInd]


nbrs = NearestNeighbors(n_neighbors=10).fit(X_tr)


X_test=X.iloc[(trInd+1):testind]

Test_dates = GenData['Date'].iloc[(trInd+1):testind]

Y_tr = GenData[Output].iloc[0:trInd]

Y_test=np.array(GenData[Output].iloc[(trInd+1):testind])



corrcoef = pacf(Y_tr,nlags=samples*7)
ImpLags = sorted(range(len(corrcoef)), key=lambda k: corrcoef[k])
 # select Top 10 lags
L=len(corrcoef)
n=3
ImpLags=np.array(ImpLags)
ImpLags1=ImpLags[ImpLags>samples]
ImpLags1=ImpLags1[0:n]
print(ImpLags1)
Y_tr=np.array(Y_tr)
Y_tr=Y_tr.reshape(len(Y_tr),1)
# ZeroPad=np.zeros((24,1))
# Y_tr =  np.concatenate((Y_tr,ZeroPad),axis=0)
LGT1=len(Y_tr)
#ImpLags1=[24,48]

ARData=np.zeros((len(Y_tr),len(ImpLags1)))
for r in range(0,len(ImpLags1)):
    ARData[ImpLags1[r]:LGT1,r]=Y_tr[0:(LGT1-(ImpLags1[r])),0]

LGT2=len(GenData)
Y=np.array(GenData[Output])
ARF = np.zeros((len(GenData),len(ImpLags1)))
for r in range(0,len(ImpLags1)):
    ARF[ImpLags1[r]:LGT2,r]=Y[0:(LGT2-(ImpLags1[r]))]

ARTest = ARF[(trInd+1):testind,:]

if ar_feat==1:
    X_tr=np.concatenate((np.array(X_tr),ARData),axis=1)
    X_test=np.concatenate((np.array(X_test),ARTest),axis=1)

    
    

ML_rgr = RandomForestRegressor(max_depth=200, random_state=42)
  # ML_rgr = ARDRegression(compute_score=True)
# ML_rgr =LinearRegression()

Y_tr=np.array(Y_tr)
Y_tr=Y_tr.reshape(len(Y_tr),1)
ML_Output=np.array(Y_tr)
ML_rgr.fit(X_tr,ML_Output.ravel())

#### predictions
ml_predictions=ML_rgr.predict(X_test)
y_pred=ml_predictions.reshape(len(ml_predictions),1)

plt.figure('RF')
plt.plot(Y_test)
plt.plot(y_pred)
plt.legend(['actuals','predictions'])

mapes =100* abs(Y_test-y_pred)/y_pred

Mape = np.mean(mapes)

print(Mape)

# plt.figure(2)
# plt.hist(mapes)

rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=1, random_state=42)


# rf =RandomForestRegressor(
#     n_estimators=300,
#     max_depth=20,
#     min_samples_split=50,
#     min_samples_leaf=25,
#     max_features='sqrt',
#     bootstrap=True,
#     oob_score=True,
#     n_jobs=-1,
#     random_state=42
# )

rf.fit(X_tr,Y_tr)

# -------------------------------
# Get predictions from all trees
# -------------------------------
all_preds = np.array([tree.predict(X_test) for tree in rf.estimators_])  # shape: (n_trees, n_test)




# -------------------------------
# Compute quantiles (e.g., 10%, 50%, 90%)
# -------------------------------
q10 = np.percentile(all_preds,2, axis=0)
q50 = np.percentile(all_preds, 50, axis=0)  # median
q90 = np.percentile(all_preds, 98, axis=0)


td=pd.to_datetime(Test_dates,dayfirst=True)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']  # list like ['#1f77b4', '#ff7f0e', ...]

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

plt.figure("RF_Q")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.plot(td, Y_test, label='Actuals', linewidth=3, linestyle='-', color=colors[0])   # solid, default first color
plt.plot(td, q10,   label='Q5',     linewidth=2, linestyle='--', color=colors[1])  # dashed, second color
plt.plot(td, q50,   label='Q50',     linewidth=2, linestyle=':',  color=colors[2])  # dotted, third color
plt.plot(td, q90,   label='Q95',     linewidth=2, linestyle='-.', color=colors[3])  # dash-dot, fourth color
plt.xlabel('Time (15 minutes)', fontsize=15, fontweight='bold')
plt.ylabel('Wind Generation (MW)', fontsize=15, fontweight='bold')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
plt.yticks(fontsize=13)
plt.xticks(rotation=40, ha='right', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend(loc='lower right',fontsize=12)
plt.show()
# plt.figure("RF_Q")
# plt.plot(td,Y_test,linewidth=3)

# plt.plot(td,q10)
# plt.plot(td,q50)
# plt.plot(td,q90)
# plt.legend(['Actuals','Q5','Q50','Q95'], fontsize=12, loc='upper center', ncol=4)
# plt.xlabel('Time (15 minutes)',  fontsize=15, fontweight='bold')
# plt.ylabel('Wind Generation (MW)',  fontsize=15, fontweight='bold')
# plt.title(alg+Horz,  fontsize=15, fontweight='bold')
# plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
# plt.yticks(fontsize=13)
# plt.xticks(rotation=40, ha='right', fontsize=13)
# plt.tight_layout()
# plt.show()


def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))


# Pinball loss
loss_q05 = pinball_loss(Y_test, q10, 0.005)
loss_q50 = pinball_loss(Y_test, q50, 0.5)
loss_q95 = pinball_loss(Y_test, q90, 0.975)

print(loss_q05,loss_q50,loss_q95)

results_Ar=np.zeros((len(Y_test),3))

results_Ar[:,0]=np.array(q10)
results_Ar[:,1]=np.array(q50)
results_Ar[:,2]=np.array(q90)

def compute_coverage(y_true, lower, upper):
    inside = (y_true >= lower) & (y_true <= upper)
    return np.mean(inside)



picp_90 = compute_coverage(Y_test, q10, q90)
print(f"Coverage for 90% PI: {picp_90:.2f}")

results_df=pd.DataFrame()

results_df['Actuals'] = np.array(Y_test)
results_df['Q5'] = np.array(q10)
results_df['Q50'] = np.array(q50)
results_df['Q95'] = np.array(q90)
results_df['Q5_L'] =loss_q05
results_df['Q50_L'] = loss_q50
results_df['Q95_L'] = loss_q95

q95 = np.array(q90)
q05 = np.array(q10)
width = q95 - q05

plt.figure(figsize=(10, 4))
plt.plot(width, label='Prediction Interval Width (90%)', color='darkorange')
plt.plot(q50)
plt.xlabel("Time Index")
plt.ylabel("Interval Width (q90 - q10)")
plt.title("Sharpness Diagram")
plt.grid(True)
plt.legend()
plt.show()


sharpness_90 = compute_sharpness(q10, q90)
print(f"Sharpness of 90% interval: {sharpness_90:.2f}")


Metrics=[]
Metrics.append(loss_q05)
Metrics.append(loss_q50)
Metrics.append(loss_q95)
Metrics.append(picp_90)
Metrics.append(sharpness_90)


j=np.array(Metrics)
j=j.reshape(1,len(j))
Matrics_Ar=pd.DataFrame(j)

Matrics_Ar.columns=['Q05','Q50','Q95','Coverage','Sharpness']





# results_df.to_csv('results_'+alg+Horz+'.csv')

Matrics_Ar.to_csv('Metrics_'+alg+Horz+'1.csv')


# y_pred_median = model.predict(X)  # your median predictions
residuals = Y_test -q50

# plt.scatter(q50, Y_test, alpha=0.5)
# plt.axhline(0, color='red', linestyle='--')
# plt.xlabel("Predicted Value")
# plt.ylabel("Residual (y - ŷ)")
# plt.title("Residuals vs. Predicted")
# plt.grid(True)
# plt.show()

# plt.hist(residuals)



# wData = pd.read_csv("C:/Users/navee/Downloads/5Loc_Weather_2024_15min.csv")

# # ---------------------------
# # 1. Generate synthetic data
# # ---------------------------
# #n_samples = 1000
# input_dim = wData.shape[1]-1
# latent_dim = 10

# # Simulated input data (e.g., wind features)
# X = wData.iloc[:,1:] #GenData[Inputs] #np.random.rand(n_samples, input_dim).astype(np.float32)

# # ---------------------------
# # 2. Define the Autoencoder
# # ---------------------------
# input_layer = tf.keras.Input(shape=(input_dim,))
# encoded = layers.Dense(1000, activation='relu')(input_layer)
# encoded = layers.Dense(latent_dim, activation='sigmoid')(encoded)

# decoded = layers.Dense(1000, activation='sigmoid')(encoded)
# decoded = layers.Dense(input_dim, activation='relu')(decoded)

# autoencoder = models.Model(inputs=input_layer, outputs=decoded)

# # Encoder model for extracting latent vectors
# encoder = models.Model(inputs=input_layer, outputs=encoded)

# # ---------------------------
# # 3. Compile and Train
# # ---------------------------
# autoencoder.compile(optimizer='adam', loss='mse')

# history = autoencoder.fit(
#     X, X,
#     epochs=1000,
#     batch_size=960*10,
#     shuffle=False,
#     validation_split=0.2
# )

# # ---------------------------
# # 4. Plot Training Loss
# # ---------------------------
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title("Training Loss (MSE)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)
# plt.show()

# decoded_rep = autoencoder.predict(X)


# def pinball_loss(y_true, y_pred, tau):
#     """
#     Computes the Pinball Loss for a given quantile level tau.
#     """
#     error = y_true - y_pred
#     loss = np.where(error >= 0, tau * error, (1 - tau) * (-error))
#     return np.mean(loss)

# import numpy as np

# # True values
# y_true = np.array([100, 120, 90, 130, 110])

# # Predicted 90th percentile values
# y_pred_90 = np.array([110, 115, 85, 125, 105])

# # Quantile level
# tau = 0.9

# # Compute pinball loss
# loss = pinball_loss(y_true, y_pred_90, tau)
# print(f"Pinball Loss at {tau*100:.0f}th percentile: {loss:.2f}")


