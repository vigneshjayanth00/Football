# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
# Importing the dataset
dataset = pd.read_csv('data.csv',encoding='iso-8859-1')
#Dataset for Positions
filter_list=['Defender']
dataset_subset=dataset[dataset.position.isin(filter_list)]
#Creating a function to negate years for age
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dataset_subset=dataset_subset.sort_values(['season'] , ascending=False)
labelencoder_X = LabelEncoder()
dataset_subset['season'] = labelencoder_X.fit_transform(dataset_subset['season'])
def func(row):
    if row['age']!=0 and row ['season'] == 0:
        return -2
    elif row['age']!=0 and row['season'] == 1:
        return -1
    else:
        return 0
dataset_subset['Age_Rank_new'] = dataset_subset.apply(func, axis=1)
dataset_subset['New_Age']=dataset_subset['age'] + dataset_subset['Age_Rank_new']


"""#Creating and Encoding Nationality
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dataset_subset=dataset_subset.sort_values(['nationality'] , ascending=False)
labelencoder_X = LabelEncoder()
dataset_subset['nationality1'] = labelencoder_X.fit_transform(dataset_subset['nationality'])
dataset_subset['New_Age']=dataset_subset['New_Age'].astype(int)
dataset_subset['nationality1']=dataset_subset['nationality1'].astype(int)"""

dataset_subset.info()
"""
cols = ["nationality1","New_Age"]
dataset_subset[cols] = dataset_subset[cols].replace({0:np.nan})
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'nan', strategy = 'mean', axis = 0)
dataset_subset=pd.DataFrame(dataset_subset)
imputer = imputer.fit(dataset_subset.iloc[:,80:81])
dataset_subset.iloc[:, 3:4] = imputer.transform(dataset_subset.iloc[:,3:4])
"""

#Creating and Encoding New_ID- New Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dataset_subset=dataset_subset.sort_values(['name'] , ascending=False)
labelencoder_X = LabelEncoder()
dataset_subset['Player_ID'] = labelencoder_X.fit_transform(dataset_subset['name'])
dataset_subset['Player_ID']=dataset_subset['Player_ID'].astype(int)


Age_Perf=dataset_subset[['New_Age','team_id','season','goals','assists','blocks',
'carries','carries_attacking_third','carries_box','carry_distance','carry_distance_attacking_third','carry_distance_box',
'passes_received','passes_received_attacking_third','tackles','tackles_successful',
'defensive_ground_duels','defensive_ground_duels_successful']]
#correlation
corr1=dataset_subset.corr()
corr2=Age_Perf.corr()


# Splitting the dataset into the Training set and Test set
Age_train=Age_correlation_performance_metrics1[Age_correlation_performance_metrics1['New_Age']!=0]
X_train = Age_train.iloc[:, 1:].values
y_train = Age_train.iloc[:, 0].values
Age_test=Age_correlation_performance_metrics1[Age_correlation_performance_metrics1['New_Age']==0]
X_test = Age_test.iloc[:, 1:].values
y_test = Age_test.iloc[:, 0].values
# Train KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X_train, y_train)
# Predict missing values' class
imputed_values = trained_model.predict(X_test)
# Join column of predicted class with their other features
X_test1 = np.hstack((imputed_values.reshape(-1,1), X_test))
# Join two feature matrices
Age_dataset=np.vstack((X_test1, Age_train))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred1 = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred1)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_train, y_pred1, normalize=True, sample_weight=None)

# Concatenate train & test
dataset=np.concatenate([y_pred, dataset_subset['Player_ID']], axis=0)

knn_impute(target=df['Age'], attributes=df.drop(['Age', 'PassengerId'], 1),
                                    aggregation_method="median", k_neighbors=10, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)



# Plot histogram using seaborn
import seaborn as sns
plt.figure(figsize=(15,8))
sns.distplot(dataset_subset.New_Age, bins =30)

#Using the predicted age variables
# Concatenate train & test
train_objs_num = len(train)
y = train['Survived']
dataset = pd.concat(objs=[train.drop(columns=['Survived']), test], axis=0)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 1))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Variables')
plt.ylabel('Age')
plt.legend()
plt.show()

#Correlation matrix
colormap = plt.cm.RdBu
plt.figure(figsize=(50,50))
plt.title('Age vs variables', y=1.05, size=15)
sns.heatmap(Age_correlation_performance_metrics1.corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)


#Correlation matrix
colormap = plt.cm.RdBu
plt.figure(figsize=(17,17))
plt.title('Age vs Performance', y=1.05, size=15)
sns.heatmap(Age_correlation_performance_metrics1.corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)

knn_impute(target=df['New_age'], attributes=df.drop(['Age', 'PassengerId'], 1),
                                    aggregation_method="median", k_neighbors=10, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)


Age_correlation_performance_metrics2=dataset_subset[['New_Age','passes_received_attacking_third','tackles','tackles_successful',
'defensive_ground_duels','defensive_ground_duels_successful']]

#Correlation matrix
colormap = plt.cm.RdBu
plt.figure(figsize=(14,13))
plt.title('Age vs Performance', y=1.05, size=15)
sns.heatmap(Age_correlation_performance_metrics2.corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import seaborn as sns
corr = dataset_subset.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#Correlation matrix
plt.matshow(dataset_subset.corr())
plt.show()

def plot_corr(dataset_subset,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = dataset_subset.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
plt.show()

#Creating and Encoding New_ID- New Dataset 1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dataset_subset=dataset_subset.sort_values(['position'] , ascending=False)
labelencoder_X = LabelEncoder()
dataset_subset['Position'] = labelencoder_X.fit_transform(dataset_subset['position'])
dataset_subset=dataset_subset.iloc[:,1:]

import statsmodels.formula.api as sm
regressor_OLS=sm.OLS(endog= Y_test,exog=X_test).fit()
regressor_OLS.summary()
dataset_subset.summary()
#
#Dataset for ranking metrics based on positions
dataset_Age_ranking=dataset.iloc[:,12:]
filter_list=['Defender', 'Midfielder']
dataset_subset=dataset_Position[dataset_Position.position.isin(filter_list)]

dataset_Position_sum=dataset_Position.groupby(['position']).rank()
dataset_Position_sum1=dataset_Position_sum.transpose()
dataset_Position_sum.to_csv('dataset_Position_sum.csv')

dataset_Position.to_csv('Data_Position.csv')
dataset_Position=dataset_Position.iloc[:,3:14]

#Filter leicester players to see their performance indicators
#Checking outgoing player profile- No stats on positions played
filter_list=['J. Maddison','W. Ndidi','B. Chilwell']
dataset_players=dataset[dataset.name.isin(filter_list)]
export_csv= dataset_players.to_csv(r'C:\Users\Home\Documents\21st Club\dataset_players.csv', index = None, header=True)
