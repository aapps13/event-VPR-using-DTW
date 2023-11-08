###########################################################################
####                           Import Modules                         #####
###########################################################################
import sys
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)

#---- Import Modules ----#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
from tqdm import tqdm
from scipy.stats import norm 

###########################################################################
####                              Set Up                              #####
###########################################################################
#---- Set up data structures ----#
linux = 0
if linux:
    print('Directories need to be defined..') # update when you know the directory
else:
    master_mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'
    master_subdtw_data_dir = 'D:/Honours/datasets/compressed_data/batch/output/'

filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}


data_dir = 'G:/Honours/spatial_compression/resolution_[100-100]/semi_long/dvs_vpr_2020-04-21-17-03-03_bias_integration.mat'
mat_data = loadmat(data_dir)['accumulation_cell']


###########################################################################
####                             Analysis                             #####
###########################################################################

m = np.zeros_like(mat_data)
d = np.zeros_like(mat_data)
r = np.zeros_like(mat_data)
residual_stats = np.zeros((mat_data.shape[0], mat_data.shape[1], 2))

ct_default = 0.1

iterable = range(mat_data.shape[0]*mat_data.shape[1])
with tqdm(iterable, desc="Computing regression for each pixel", unit="item") as progress:
    for i in range(mat_data.shape[0]):
        for j in range(mat_data.shape[1]):
            data = mat_data[i,j].T

            #---- Linear Regression ----#
            x = np.arange(0, len(data), 1).reshape(-1, 1)
            y = data

            regr = linear_model.LinearRegression()
            regr.fit(x, y)
            y_pred = regr.predict(x)
            residuals = y_pred - y
            rmse = metrics.mean_squared_error(y, y_pred, squared=False)

            #---- Gaussian Fit ----#
            mu, std = norm.fit(residuals)

            #---- Store the values ----#
            m[i,j] = regr.coef_[0]
            d[i,j] = regr.intercept_[0]
            r[i,j] = rmse
            residual_stats[i,j,:] = [mu, std]

            progress.set_description(f"Processing pixel ({i},{j})")
            progress.update(1)

scale = np.median(ct_default * r) / r
bias = - scale * m

print(scale[0,0])
print(bias[0,0])

fig, ax = plt.subplots(1,2)
ax[0].imshow(scale)
ax[1].imshow(bias)
plt.show()
exit()

###########################################################################
####                             Plotting                             #####
###########################################################################
#---- Colours ----#
green_sea = '#16a085'
emerald = '#2ecc71'
peter_river = '#3498db'
amythest = '#9b59b6'
carrot = '#e67e22'
pomegranate = '#c0392b'

#---- Plot Params ----#
suptitle_weight = 'bold'
suptitle_size = 16
title_weight = 'bold'
title_size = 14
label_weight = 'normal'
label_size = 12

print(f'Pixel: ({i},{j}) \t Count: {len(data)} \nCoefficient: {regr.coef_} \t Intercept: {regr.intercept_} \nmu: {mu} \t std: {std}')

fig, ax = plt.subplots(1,3, figsize=(15,5))
#---- Linear Fit ----#
ax[0].set_title('Data with Linear Model', fontweight=title_weight, fontsize=title_size)
ax[0].plot(x, y, label='Actual Data', color=peter_river)
ax[0].plot(x, y_pred, label='Linear Model', color=pomegranate)
ax[0].legend(fontsize=label_size)
ax[0].grid(which='both')
ax[0].set_xlabel('Event (k)', fontweight=label_weight, fontsize=label_size)
ax[0].set_ylabel('Accumulated Polarity (count)', fontweight=label_weight, fontsize=label_size)

#---- Residuals ----#
ax[1].set_title('Residuals of Linear Model', fontweight=title_weight, fontsize=title_size)
ax[1].plot(x, residuals, label='Residuals', color=peter_river)
ax[1].legend(fontsize=label_size)
ax[1].grid(which='both')
ax[1].set_xlabel('Event (k)', fontweight=label_weight, fontsize=label_size)
ax[1].set_ylabel('Accumulated Polarity (count)', fontweight=label_weight, fontsize=label_size)

#---- Histogram with Gaussian ----#
ax[2].set_title('Distribution of Residuals', fontweight=title_weight, fontsize=title_size)
ax[2].hist(residuals, bins=100)
ax[2].grid(which='both')
ax[2].set_xlabel('Residual Value (count)', fontweight=label_weight, fontsize=label_size)
ax[2].set_ylabel('Bin Count', fontweight=label_weight, fontsize=label_size)
plt.show()