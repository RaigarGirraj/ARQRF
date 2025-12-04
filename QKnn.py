# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 22:37:28 2025

@author: Girraj
"""


# import torch
# print(torch.__version__)

import pandas as pd
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

import os

alg = 'KNN'
Horz = '24H'

samples = 24*4 ## number of samples for day ahead (forecast horizon)

Test_H =7*samples ## 7 days moving window forecast
plt.rcParams.update({'font.size': 18})  # Set font size globally


def knn_regression(X_tr,Y_tr, X_test, k):

    nbrs = NearestNeighbors(n_neighbors=k).fit(X_tr)
    distances, indices = nbrs.kneighbors(X_test)

    y_pred = np.zeros((len(X_test),3))
    j=0
    for idx in indices:
        y_neighbors = Y_tr[idx]
        y_pred[j,0]=np.min(y_neighbors)
        y_pred[j,1]=np.mean(y_neighbors)
        y_pred[j,2]=np.max(y_neighbors)
        j=j+1

    return np.array(y_pred)

# Gaussian Kernel Function
# -------------------------------
def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

# -------------------------------
# KW-KNN Regression Function
# -------------------------------
def kw_knn_regression(X_train, y_train, X_test, k, bandwidth=1.0):
    # Find K nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_train)
    distances, indices = nbrs.kneighbors(X_test)

    y_pred = []

    # For each test sample
    for dist, idx in zip(distances, indices):
        y_neighbors = y_train[idx]
        weights = gaussian_kernel(dist, bandwidth)

        # Weighted average
        weighted_sum = np.sum(weights * y_neighbors)
        normalization = np.sum(weights)
        prediction = weighted_sum / normalization
        y_pred.append(prediction)

    return np.array(y_pred)

def kw_knn_quantiles(X_tr, Y_tr, X_test,k, quantiles=[0.1, 0.5, 0.9], bandwidth=1):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_tr)
    distances, indices = nbrs.kneighbors(X_test)

    all_quantiles = []

    for dist, idx in zip(distances, indices):
        y_neighbors = np.array(Y_tr[idx])
        weights = gaussian_kernel(dist, bandwidth)

        # Sort y-values and weights
        sorted_idx = np.argsort(y_neighbors)
        y_sorted = y_neighbors[sorted_idx]
        w_sorted = weights[sorted_idx]
        w_norm = w_sorted / np.sum(w_sorted)

        # Build empirical CDF
        cdf = np.cumsum(w_norm)

        # Interpolate to get quantile estimates
        quantile_func = interp1d(cdf, y_sorted, bounds_error=False,
                                 fill_value=(y_sorted[0], y_sorted[-1]))
        q_values = [quantile_func(q) for q in quantiles]
        all_quantiles.append(q_values)

    return np.array(all_quantiles)


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

GenData = pd.read_csv("D:/Movies/CaISO_GenData_2024_15min.csv")

# Wdata = pd.read_csv("C:/Users/navee/Downloads/5Loc_Weather_2024_15min.csv")

Output = 'Wind+Curtail'

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





ar_feat=1


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
k=500
y_pred_Knn = knn_regression(X_tr,Y_tr, X_test, k)

y_pred_Knn_kw=kw_knn_regression(X_tr, Y_tr, X_test, k, bandwidth=1.0)

y_pred_Knn_Q = kw_knn_quantiles(X_tr, Y_tr, X_test,k, quantiles=[0.1, 0.5, 0.9], bandwidth=1)


plt.figure("KNN")
plt.plot(Y_test)
# plt.plot(y_pred_Knn)


for j in range(0,y_pred_Knn.shape[1]):
    plt.plot(y_pred_Knn[:,j])
    
plt.legend(['Actuals','Min','Avg','Max'])

q10=y_pred_Knn[:,0]
q50=y_pred_Knn[:,1]

q90=y_pred_Knn[:,2]


td=pd.to_datetime(Test_dates,dayfirst=True)

plt.figure("RF_Q")
plt.plot(td,Y_test,linewidth=3)

plt.plot(td,q10)
plt.plot(td,q50)
plt.plot(td,q90)
plt.legend(['Actuals','Q5','Q50','Q95'])
plt.xlabel('Time(15 minutes)')
plt.ylabel('Wind_Generation(MW)')
plt.title(alg+Horz)



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





