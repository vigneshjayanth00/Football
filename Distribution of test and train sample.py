# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:44:18 2020

@author: Home
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#importing data
data5= pd.read_csv(r'C:\Users\Home\Documents\OptaPro Forum20012020\Updated Data\LB SUMMARY\Final-07-05-2020\Final.csv')
data=data5
#Data manipulation
data['duration']=(data['duration']/25)
data['forward_run_strategy_id']=data['run_strategy_id']-data['hs_run_strategy_id']
data['did_it_break1']=np.where(data['did_it_break']==True,1,0)
del data['did_it_break']
del data['run_strategy_id']
del data['match_id']
data1=data[data['duration']>8]

#Removing the duration field as it may distort the model
data2=data1.iloc[:,5:]

# split into input (X) and output (Y) variables
X = data2.iloc[:,0:5]
Y = data2.iloc[:,5]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.3, random_state = 3)

#Plotting the image

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.kdeplot(y_train, ax=ax, label='Train', lw=5, alpha=0.6);
    sns.kdeplot(y_test, ax=ax, label='Test', lw=5, alpha=0.6);
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=20);
    
###Test the sample size and resample to see the distribution of the sample    
SAMPLE_SIZE = 50000
N_BINS = 300

# Obtain `N_BINS` equal frequency bins, in other words percentiles
step = 100 / N_BINS
test_percentiles = [
    np.percentile(y_test, q, axis=0)
    for q in np.arange(start=step, stop=100, step=step)
]

# Match each observation in the training set to a bin
train_bins = np.digitize(y_train, test_percentiles)

# Count the number of values in each training set bin
train_bin_counts = np.bincount(train_bins)

# Weight each observation in the training set based on which bin it is in
weights = 1 / np.array([train_bin_counts[x] for x in train_bins])

# Make the weights sum up to 1
weights_norm = weights / np.sum(weights)

np.random.seed(0)
sample = np.random.choice(train, size=SAMPLE_SIZE, p=weights_norm, replace=False)
