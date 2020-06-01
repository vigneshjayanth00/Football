# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:15:44 2020

@author: Home
"""

#import libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

from sklearn.datasets import load_digits


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
del data1['overload_strategy_id']
del data1['hs_run_strategy_id']
del data1['spread_strategy_id']
#Removing the duration field as it may distort the model
data2=data1.iloc[:,4:]

# split into input (X) and output (Y) variables
X = data2.iloc[:,0:6]
Y = data2.iloc[:,6]

del data1['spread_strategy_id']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.35, random_state = 3)

model = DecisionTreeClassifier(criterion='entropy',max_depth=1)
AdaBoost = AdaBoostClassifier(base_estimator= model,n_estimators=400,learning_rate=0.75,random_state=0)

#AdaBoost = AdaBoostClassifier(n_estimators=400,learning_rate=0.01,algorithm='SAMME')

model=AdaBoost.fit(X_train,y_train)
y_test_pred=model.predict(X_test)
y_train_pred=model.predict(X_train)

train_pred = AdaBoost.score(X_train,y_train)
test_pred = AdaBoost.score(X_test,y_test)

print('The train set accuracy is: ',train_pred*100,'%')
print('The test set accuracy is: ',test_pred*100,'%')


#Print confusion matrix for both test and train sets
cm_test = confusion_matrix(y_test, y_test_pred)
cm_train = confusion_matrix(y_train, y_train_pred)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# keep probabilities for the positive outcome only
Adaboost_probs = model.predict_proba(X_test)
Adaboost_probs = Adaboost_probs[:, 1]

#Predict Probabilities- of both test and train sets
Adaboost_train_probs = model.predict_proba(X_train)
Adaboost_train_probs = Adaboost_train_probs[:, 1]
Adaboost_test_probs = model.predict_proba(X_test)
Adaboost_test_probs = Adaboost_test_probs[:, 1]

#Check ROC/AUC Score for test dataset
ns_auc = roc_auc_score(y_test, ns_probs)
Ada_auc = roc_auc_score(y_test, Adaboost_probs)

#testing for both test and training set
Ada_test_auc = roc_auc_score(y_test, Adaboost_test_probs)
Ada_train_auc = roc_auc_score(y_train, Adaboost_train_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('AdaBoost: ROC AUC=%.3f' % (Ada_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, Adaboost_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='AdaBoost')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

#######################################################################################

from sklearn.metrics import auc
lr_precision, lr_recall, _ = precision_recall_curve(y_test, Adaboost_probs)
lr_f1, lr_auc = f1_score(y_test, y_test_pred), auc(lr_recall, lr_precision)
# summarize scores
print('Adaboost: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Adaboost')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()



#######################################################################################
#Feature Importance
print(model.feature_importances_)
Ada_support_feature = X.columns.tolist()
Ada_support_feature_df = pd.DataFrame({'Feature':Ada_support_feature, 'Feature Importance': model.feature_importances_})

# plot
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

scores_ada = cross_val_score(model, X, Y, cv=6)
scores_ada.mean()

score = []
for depth in [1,2,10] : 
    reg_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth))
    scores_ada = cross_val_score(reg_ada, X, Y, cv=6)
    score.append(scores_ada.mean())
#############################################################################################
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = AdaBoostClassifier(base_estimator= model,n_estimators=400, learning_rate=learning_rate,random_state=0)
    gb_clf.fit(X_train,y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))