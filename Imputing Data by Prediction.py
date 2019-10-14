# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
# Importing the dataset
dataset_subset = pd.read_csv('21st-club_junior-analyst_data.csv',encoding='iso-8859-1')

#Dataset Statistics
des=dataset_subset.describe()

#Creating and Encoding Nationality as a new variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dataset_subset=dataset_subset.sort_values(['nationality'] , ascending=False)
labelencoder_X = LabelEncoder()
dataset_subset['nationality1'] = labelencoder_X.fit_transform(dataset_subset['nationality'])
#Creating and Encoding Player ID
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dataset_subset=dataset_subset.sort_values(['name'] , ascending=False)
labelencoder_X = LabelEncoder()
dataset_subset['Player_ID'] = labelencoder_X.fit_transform(dataset_subset['name'])
dataset_subset['Player_ID']=dataset_subset['Player_ID'].astype(int)

#Encoding Position as a new variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dataset_subset=dataset_subset.sort_values(['position'] , ascending=False)
labelencoder_X = LabelEncoder()
dataset_subset['position1'] = labelencoder_X.fit_transform(dataset_subset['position'])
dataset_subset['position1']=dataset_subset['position1'].astype(int)

#Encoding Season as a new variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dataset_subset=dataset_subset.sort_values(['season'] , ascending=False)
labelencoder_X = LabelEncoder()
dataset_subset['season1'] = labelencoder_X.fit_transform(dataset_subset['season'])

Age_Perf=dataset_subset[['age','nationality1','position1','position_am_perc','position_cb_perc','position_cf_perc','position_cm_perc','position_dm_perc','position_gk_perc',
'position_lb_perc','position_lw_perc','position_rb_perc','position_rw_perc',
'Player_ID','team_id','season1','goals','assists','blocks',
'carries','carries_attacking_third','carries_box','carry_distance','carry_distance_attacking_third','carry_distance_box',
'passes_received','passes_received_attacking_third','tackles','tackles_successful',
'defensive_ground_duels','defensive_ground_duels_successful']]

# Splitting the dataset into the Training set and Test set
Age_train=Age_Perf[Age_Perf['age']!=0]
X_train = Age_train.iloc[:, 1:].values
Y_train = Age_train.iloc[:, 0:1].values
Age_test=Age_Perf[Age_Perf['age']==0]
X_test = Age_test.iloc[:, 1:].values
Y_test = Age_test.iloc[:, 0:1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train1 = sc.fit_transform(X_train)
X_test1 = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train1, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test1)
y_pred1=classifier.predict(X_train1)
y_pred1=y_pred1.astype(int)

#Measuring accuracy between Training set and predicted values on training set
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred1, Y_train, normalize=True, sample_weight=None)

# Join column of predicted class with their other features
Age_test = np.hstack((X_test,y_pred.reshape(-1,1)))
Age_train = np.hstack((X_train,Y_train.reshape(-1,1)))

# Join two feature matrices
Age_dataset=np.vstack((Age_test, Age_train))
Age_dataset=pd.DataFrame(Age_dataset)

#Adding columns to the dataset
Age_dataset.columns=['nationality1','position1','position_am_perc','position_cb_perc','position_cf_perc','position_cm_perc','position_dm_perc','position_gk_perc',
'position_lb_perc','position_lw_perc','position_rb_perc','position_rw_perc',
'Player_ID','team_id','season1','goals','assists','blocks',
'carries','carries_attacking_third','carries_box','carry_distance','carry_distance_attacking_third','carry_distance_box',
'passes_received','passes_received_attacking_third','tackles','tackles_successful',
'defensive_ground_duels','defensive_ground_duels_successful','age']

#Correcting the Age across seasons
Age_dataset=Age_dataset.sort_values(['Player_ID'] , ascending=False)

def func(row):
    if row['age']!=0 and row ['season1'] == 0:
        return -2
    elif row['age']!=0 and row['season1'] == 1:
        return -1
    else:
        return 0
Age_dataset['Age_Rank_new'] = Age_dataset.apply(func, axis=1)
Age_dataset['New_Age']=Age_dataset['age'] + Age_dataset['Age_Rank_new']

del Age_dataset['Age_Rank_new']
del Age_dataset['age']

#Predicting Nationality
# Splitting the dataset into the Training set and Test set
Age_train1=Age_dataset[Age_dataset['nationality1']!=0]
X_train1 = Age_train1.iloc[:, 1:].values
Y_train1 = Age_train1.iloc[:,0:1].values
Age_test1=Age_dataset[Age_dataset['nationality1']==0]
X_test1 = Age_test1.iloc[:, 1:].values
Y_test1 = Age_test1.iloc[:, 0:1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train1)
X_test2 = sc.transform(X_test1)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train2, Y_train1)

# Predicting the Test set results
y_pred = classifier.predict(X_test2)
y_pred1=classifier.predict(X_train2)
y_pred1=y_pred1.astype(int)
#Measuring accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred1, Y_train1, normalize=True, sample_weight=None)
# Join column of predicted class with their other features
Age_Nat_test = np.hstack((X_test1,y_pred.reshape(-1,1)))
Age_Nat_train = np.hstack((X_train1,Y_train1.reshape(-1,1)))
# Join two feature matrices
Age_dataset2=np.vstack((Age_Nat_test, Age_Nat_train))

#Adding Columns to the Dataset
Age_dataset2=pd.DataFrame(Age_dataset2)
Age_dataset2.columns=['position1','position_am_perc','position_cb_perc','position_cf_perc','position_cm_perc','position_dm_perc','position_gk_perc',
'position_lb_perc','position_lw_perc','position_rb_perc','position_rw_perc',
'Player_ID','team_id','season1','goals','assists','blocks',
'carries','carries_attacking_third','carries_box','carry_distance','carry_distance_attacking_third','carry_distance_box',
'passes_received','passes_received_attacking_third','tackles','tackles_successful',
'defensive_ground_duels','defensive_ground_duels_successful','age','nationality1']


