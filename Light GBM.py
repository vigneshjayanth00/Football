# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:58:19 2020

@author: VigneshJayanth
"""

import lightgbm as lgb
from sklearn import metrics


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
data2=data1.iloc[:,4:]

# split into input (X) and output (Y) variables
X = data2.iloc[:,0:5]
Y = data2.iloc[:,5]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.3, random_state = 3)

#Fitting the Light GBM Classifier
def auc2(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict(train)),
                            metrics.roc_auc_score(y_test,m.predict(test)))

lg = lgb.LGBMClassifier(silent=False)
param_dist = {"max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]
             }
grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
grid_search.fit(X_train,y_train)
#####After this step: type grid_search.best_estimator_ in the ipython console for the best estimator
d_train = lgb.Dataset(X_train, label=y_train)
params = {"max_depth": 25, "learning_rate" : 0.001, "num_leaves": 200,  "n_estimators": 300}

# Without Categorical Features
model2 = lgb.train(params, d_train)

y_test_pred=model2.predict(X_test)
y_train_pred=model2.predict(X_train)

#convert into binary values
for i in range(0,len(y_test_pred)):
    if y_test_pred[i]>=.5:       # setting threshold to .5
       y_test_pred[i]=1
    else:  
       y_test_pred[i]=0

for i in range(0,len(y_train_pred)):
    if y_train_pred[i]>=.5:       # setting threshold to .5
       y_train_pred[i]=1
    else:  
       y_train_pred[i]=0


#Confusion matrix
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_test_pred)
cm_train = confusion_matrix(y_train, y_train_pred)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy_test=accuracy_score(y_test_pred,y_test)
accuracy_train=accuracy_score(y_train_pred,y_train)

print('The train set accuracy is: ',accuracy_train*100,'%')
print('The test set accuracy is: ',accuracy_test*100,'%')

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# keep probabilities for the positive outcome only
lgb_boost_probs = model2.predict(X_test)

#Predict Probabilities- of both test and train sets
lgb_boost_train_probs = model2.predict(X_train)
lgb_boost_test_probs = model2.predict(X_test)

#Check ROC/AUC Score for test dataset
ns_auc = roc_auc_score(y_test, ns_probs)
lgb_auc = roc_auc_score(y_test, lgb_boost_probs)

#testing for both test and training set
lgb_test_auc = roc_auc_score(y_test, lgb_boost_test_probs)
lgb_train_auc = roc_auc_score(y_train, lgb_boost_train_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('lgb_Boost: ROC AUC=%.3f' % (lgb_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lgb_boost_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='lgb_Boost')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

