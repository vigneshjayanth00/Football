# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
# Importing the dataset
dataset = pd.read_csv('data.csv',encoding='iso-8859-1')
#Use Feature Engineering for first import steps and cleaning
#test_set for Age
dataset_test=dataset[dataset['age']!=0]
filter_list=['Defender', 'Midfielder']
dataset_test=dataset_test[dataset_test.position.isin(filter_list)]

#Creating and Encoding New_ID- New Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dataset_test=dataset_test.sort_values(['name'] , ascending=False)
labelencoder_X = LabelEncoder()
dataset_test['Player_ID'] = labelencoder_X.fit_transform(dataset_test['name'])
dataset_test['Player_ID']=dataset_test['Player_ID'].astype(int)

#Creating a function to negate years for age
dataset_test['Age_Rank'] = dataset_test.groupby('Player_ID')['season'].rank(ascending=False)

def func(row):
    if row['Age_Rank'] == 3:
        return -2
    elif row['Age_Rank'] == 2:
        return -1
    else:
        return 0
dataset_test['Age_Rank_new'] = dataset_test.apply(func, axis=1)
dataset_test['New_Age']=dataset_test['age'] + dataset_test['Age_Rank_new']

dataset_test=dataset_test.drop(['Age_Rank','Age_Rank_new','player_id','Player_ID','age',
                                ], axis=1)
dataset_test1=dataset_test[['New_Age','goals','assists']]

#Dataset to check position correlation to positions
#Whole dataset
filter_list=['Defender', 'Midfielder']
Position_correlation=dataset[dataset.position.isin(filter_list)]
Position_correlation=Position_correlation[['position','position_am_perc','position_cb_perc','position_cf_perc','position_cm_perc','position_dm_perc',
'position_gk_perc','position_lb_perc','position_lw_perc','position_rb_perc','position_rw_perc']]

# Recoding value from numeric to string
Position_correlation['position'].replace({'Defender':1, 'Midfielder':2, 'Attacker':3}, inplace= True)

#Correlation Plot
colormap = plt.cm.RdBu
plt.figure(figsize=(14,13))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(Position_correlation.corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)
#Test set, Age!=0
Position_correlation_New_age=dataset_test[['position','position_am_perc','position_cb_perc','position_cf_perc','position_cm_perc','position_dm_perc',
'position_gk_perc','position_lb_perc','position_lw_perc','position_rb_perc','position_rw_perc']]




dataset_test3=dataset_test[['New_Age','goals','assists','minutes_played','passes_collected','blocks',
'carries','carries_attacking_third','carries_box','carry_distance','carry_distance_attacking_third','carry_distance_box',
'passes_received','passes_received_attacking_third','passes_received_box','tackles','tackles_successful',
'defensive_ground_duels','defensive_ground_duels_successful','defensive_aerial_duels','defensive_aerial_duels_successful']]
info=dataset_test1.describe()

plt.matshow(dataset_test1.corr())
plt.show()



describe=dataset_test.describe()
dataset_test1['New_age']=dataset_test.groupby('New_Age').mean()




dataset_test1=dataset_test.iloc[:,2:]

#Correlation matrix
colormap = plt.cm.RdBu
plt.figure(figsize=(14,13))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(dataset_test2.corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)




#export
dataset_test1.to_csv('Age_Performance_Metrics.csv')


#Creating X and Y variable
X_test = dataset_test1.iloc[:, :-1].values
Y_test = dataset_test1.iloc[:,-1].values

#Standardizing values using scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_test=sc_x.fit_transform(X_test)

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(X_test,Y_test)

#Predicting
y_pred=Regressor.predict(X_test)
y_pred=y_pred.astype(int)
Y_test= Y_test.astype(int)


#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
regressor_OLS=sm.OLS(endog= Y_test,exog=X_test).fit()
regressor_OLS.summary()






dataset_test=dataset_test.iloc[:,8:]


#InfoStats from the Dataset
Info=dataset.info()
#Describe Dataset
Describe=dataset.describe()

#Correlation between age and goals
X_test=dataset_test.iloc[:,:-1]
Y_test=dataset_test.iloc[:,-1]
Y_test= Y_test.astype(int)







