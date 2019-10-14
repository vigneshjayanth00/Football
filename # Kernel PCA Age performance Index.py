# Kernel PCA

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# Subsetting the dataset
Age_dataset3=Age_dataset2[Age_dataset2['position1']==3]

#Features
features=['goals','assists','blocks',
'carries','carries_attacking_third','carries_box','carry_distance','carry_distance_attacking_third','carry_distance_box',
'passes_received','passes_received_attacking_third','tackles','tackles_successful',
'defensive_ground_duels','defensive_ground_duels_successful']

# Seperating the Features
X = Age_dataset3.loc[:, features].values
# Separating out the target
Y = Age_dataset3.iloc[:, Age_dataset2.columns == 'age'].values

#Correlation check
Corr_Team=Age_dataset3.corr()

#Correlation matrix
colormap = plt.cm.RdBu
plt.figure(figsize=(50,50))
plt.title('Age vs variables', y=1.05, size=15)
sns.heatmap(Age_dataset3.corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1 = sc.fit_transform(X)
X1=pd.DataFrame(X1)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
X_new = pca.fit_transform(X1)
explained_variance = pca.explained_variance_ratio_

# Dump components relations with features:
principalDf = pd.DataFrame(data = pca.components_
                 , columns=list(X1.columns))

Age_Nat_test = np.hstack((X,Y))


#Creating the Age Performance Index

Age_dataset3['Age_Performance_Index']=((Age_dataset3['goals'])+(Age_dataset3['assists'])+(Age_dataset3['blocks'])
+(Age_dataset3['carries'])+(Age_dataset3['carries_attacking_third'])+(Age_dataset3['carries_box'])+(Age_dataset3['carry_distance'])+(Age_dataset3['carry_distance_attacking_third'])+(Age_dataset3['carry_distance_box'])
+(Age_dataset3['passes_received'])+(Age_dataset3['passes_received_attacking_third'])+(Age_dataset3['tackles'])+(Age_dataset3['tackles_successful'])
+(Age_dataset3['defensive_ground_duels'])+(Age_dataset3['defensive_ground_duels_successful']))*.23

Age_dataset4 = pd.merge(Age_dataset3[['Age_Performance_Index','season1','Player_ID','age','team_id','nationality1']],dataset_subset[['name', 'team', 'competition','Player_ID']],how='left',
                 on=['Player_ID'])

Age_dataset5 = pd.merge(Age_dataset4,Age_dataset2[['nationality']],how='left',
                 on=['nationality1'])


# Importing the dataset
dataset= pd.read_csv('21st-club_junior-analyst_data.csv',encoding='iso-8859-1')
filter_list=[['B. Chilwell','W. Ndidi','J. Maddison']]
dataset=dataset[dataset.name.isin(filter_list)]

Age_dataset4.to_csv(r'C:\Users\Home\Documents\21st Club\Age_dataset4.csv')
dataset_subset.to_csv(r'C:\Users\Home\Documents\21st Club\dataset_subset.csv')
dataset.to_csv(r'C:\Users\Home\Documents\21st Club\dataset.csv')



