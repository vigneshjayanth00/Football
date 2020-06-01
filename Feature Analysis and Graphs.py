# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:37:35 2020

@author: Home
"""

#Visualizing Classes

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(Y,label="Sum")

plt.show()

#Describing the dataset features
Value_Analysis=pd.DataFrame(X.describe().transpose())


#Identifyinf correlation between X variables (features)
corr = X.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(10, 10))
sns.heatmap(corr,annot=True,square=True)


#Identifyinf correlation between X and Y variable (target)
Break_corr = X.join(Y).corr()

mask = np.zeros((6,6))
mask[:12,:]=1

plt.figure(figsize=(10, 10))
with sns.axes_style("white"):
    sns.heatmap(Break_corr, annot=True,square=True,mask=mask)
    
    
    
#Distribution Analysis
    
sns.distplot(X[["overload_strategy_id"]])
plt.title("Overload Strategy Distribution with KDE")

sns.distplot(X[["spread_strategy_id"]])
plt.title("Spread Strategy Distribution with KDE")

sns.distplot(X[["pocket_strategy_id"]])
plt.title("Pocket Strategy Distribution with KDE")

sns.distplot(X[["hs_run_strategy_id"]])
plt.title("Half-Space Run Strategy Distribution with KDE")

sns.distplot(X[["forward_run_strategy_id"]])
plt.title("Forward Run Strategy Distribution with KDE")

sns.distplot(X[["duration"]])
plt.title("Duration Distribution with KDE")


#Pie Chart of positive and negative outcomes

#https://matplotlib.org/gallery/pie_and_polar_charts/pie_features.html

source_counts =pd.DataFrame(Y.value_counts()).reset_index()
source_counts.columns =["Labels","Broken"]

fig1, ax1 = plt.subplots()
explode = (0, 0.15)
ax1.pie(source_counts["Broken"], explode=explode, labels=source_counts["Labels"], autopct='%1.1f%%',
        shadow=True, startangle=70)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Line Broken Percentage")
plt.show()

#Checking every feature distribution vs target- overloads strategy
plt.figure(figsize=(10,6))
sns.distplot(data2[data2["did_it_break1"] == 0]['overload_strategy_id'], color = 'r',label='Line Broken=0',kde=False)
sns.distplot(data2[data2["did_it_break1"] == 1]['overload_strategy_id'], color = 'b',label='Line Rroken=1',kde=False)
plt.legend()
plt.title("Overloads Distribution")


plt.figure(figsize=(10,6))
sns.distplot(data2[data2["did_it_break1"] == 0]['spread_strategy_id'], color = 'r',label='Line Broken=0',kde=False)
sns.distplot(data2[data2["did_it_break1"] == 1]['spread_strategy_id'], color = 'b',label='Line Rroken=1',kde=False)
plt.legend()
plt.title("Spreads Distribution")


plt.figure(figsize=(10,6))
sns.distplot(data2[data2["did_it_break1"] == 0]['duration'], color = 'r',label='Line Broken=0',kde=False)
sns.distplot(data2[data2["did_it_break1"] == 1]['duration'], color = 'b',label='Line Rroken=1',kde=False)
plt.legend()
plt.title("Duration Distribution")


plt.figure(figsize=(10,6))
sns.distplot(data2[data2["did_it_break1"] == 0]['pocket_strategy_id'], color = 'r',label='Line Broken=0',kde=False)
sns.distplot(data2[data2["did_it_break1"] == 1]['pocket_strategy_id'], color = 'b',label='Line Rroken=1',kde=False)
plt.legend()
plt.title("Central Pocket Distribution")


plt.figure(figsize=(10,6))
sns.distplot(data2[data2["did_it_break1"] == 0]['forward_run_strategy_id'], color = 'r',label='Line Broken=0',kde=False)
sns.distplot(data2[data2["did_it_break1"] == 1]['forward_run_strategy_id'], color = 'b',label='Line Rroken=1',kde=False)
plt.legend()
plt.title("Forward Run Distribution")


plt.figure(figsize=(10,6))
sns.distplot(data2[data2["did_it_break1"] == 0]['hs_run_strategy_id'], color = 'r',label='Line Broken=0',kde=False)
sns.distplot(data2[data2["did_it_break1"] == 1]['hs_run_strategy_id'], color = 'b',label='Line Rroken=1',kde=False)
plt.legend()
plt.title("Half Space Run Distribution")

# calculate the Pearson's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# calculate spearman's correlation
corr, _ = spearmanr(X[["spread_strategy_id"]], X[["hs_run_strategy_id"]])
print('Spearmans correlation: %.3f' % corr)

# calculate Pearson's correlation
seed(1)
corr = pearsonr(X[["spread_strategy_id"]], X[["hs_run_strategy_id"]])
print('Pearsons correlation: %.3f' % corr)
