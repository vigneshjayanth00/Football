# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:31:03 2020

@author: Home
"""


#Since it's a gradient boosting algorithm, we'll not resample it using bagging or bootstrapping methods
#First we'll have to study the Distribution of the testing and training sample, if it follows a gaussian or
#normal distribution then we could use a 


#importing data
data5= pd.read_csv(r'C:\Users\Home\Documents\OptaPro Forum20012020\Updated Data\LB SUMMARY\Final-07-05-2020\Final.csv')
data=data5
#Data manipulation
data['duration']=(data['duration']/25)
data['forward_run_strategy_id']=data['run_strategy_id']-data['hs_run_strategy_id']
data['did_it_break1']=np.where(data['did_it_break']==True,1,0)
del data['did_it_break']
del data['run_strategy_id']
del data['overload_strategy_id']
del data['match_id']
data1=data[data['duration']>8]

#Removing the duration field as it may distort the model
data2=data1.iloc[:,4:]

# split into input (X) and output (Y) variables
del data2['pocket_strategy_id']
X = data2.iloc[:,0:5]
Y = data2.iloc[:,5]

del X['overload_strategy_id']
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.35, random_state = 3)

###########################################################################################
#XGBoost- when data is normalized
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# Parameter Tuning
model = xgb.XGBClassifier()
param_dist = {"max_depth": [3,4,5,6,7,8],
              "min_child_weight" : [1,3,6,9,10],
              "n_estimators": [200,100,50],
              "learning_rate": [0.05, 0.1,0.16,0.2,0.5,0.75,0.01],}
grid_search = GridSearchCV(model, param_grid=param_dist, cv = 5, 
                                   verbose=10, n_jobs=-1)
grid_search.fit(X_train, y_train)
#To search for the best estimator use grid_search.best_estimator_ in the Ipython Console
model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.003, max_delta_step=0, max_depth=3,
              min_child_weight=4, missing=None, n_estimators=250, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train,y_train,early_stopping_rounds=10,eval_metric=["error", "logloss","auc","map"], eval_set=eval_set, verbose=True)

# Predict labels for train set and assess accuracy
y_train_pred = model.predict(X_train)
# Predict labels for test set and assess accuracy
y_test_pred = model.predict(X_test)

#Check Accuracy Score of both training and test sets
y_test_accuracy = accuracy_score(y_test, y_test_pred)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print('Training accuracy: ', y_train_accuracy)
print('Test accuracy: ', y_test_accuracy)

#Compute F1 score due to class imbalance
from sklearn.metrics import f1_score
y_test_f1_score=f1_score(y_test, y_test_pred, average='macro')
y_train_f1_score=f1_score(y_train, y_train_pred, average='macro')
f1_score=f1_score(y_train, y_train_pred, average='macro')

print('Training f1_score: ', y_train_f1_score)
print('Test f1_score: ', y_test_f1_score)

#Compute Cohen's Kappa score due to class imbalance
from sklearn.metrics import cohen_kappa_score
y_test_ck_score=cohen_kappa_score(y_test, y_test_pred)
y_train_ck_score=cohen_kappa_score(y_train, y_train_pred)

print('Training cohen_kappa_score: ', y_train_ck_score)
print('Test cohen_kappa_score: ', y_test_ck_score)

#Print confusion matrix for both test and train sets
cm_test = confusion_matrix(y_test, y_test_pred)
cm_train = confusion_matrix(y_train, y_train_pred)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# keep probabilities for the positive outcome only
xG_probs = model.predict_proba(X_test)
xG_probs = xG_probs[:, 1]

#Predict Probabilities- of both test and train sets
xG_train_probs = model.predict_proba(X_train)
xG_train_probs = xG_train_probs[:, 1]
xG_test_probs = model.predict_proba(X_test)
xG_test_probs = xG_test_probs[:, 1]

#Check ROC/AUC Score for test dataset
ns_auc = roc_auc_score(y_test, ns_probs)
xG_auc = roc_auc_score(y_test, xG_probs)

#testing for both test and training set
xG_test_auc = roc_auc_score(y_test, xG_test_probs)
xG_train_auc = roc_auc_score(y_train, xG_train_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('xG Boost: ROC AUC=%.3f' % (xG_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, xG_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='xG Boost')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


#Feature Importance
print(model.feature_importances_)
# plot
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')

from xgboost import plot_importance
# plot feature importance
plot_importance(model)
pyplot.show()

#############################################################################################
# retrieve performance metrics
results = m.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()

##############################################\\\\\\\\\\\\\\\\####################################
#Drop first column from features set as it has zero feature importance- Dimensionality Reduction
##############################################\\\\\\\\\\\\\\\\####################################
##################################################################################################

X_train_scaled=X_train_scaled.iloc[:,1:5]
X_test_scaled=X_test_scaled.iloc[:,1:5]

# Parameter Tuning
model = xgb.XGBClassifier()
param_dist = {"max_depth": [3,4,5,6,7,8],
              "min_child_weight" : [1,3,6,9,10],
              "n_estimators": [200],
              "learning_rate": [0.05, 0.1,0.16,0.2,0.3,0.9,0.01],}
grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, 
                                   verbose=10, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
#To search for the best estimator use grid_search.best_estimator_ in the Ipython Console
model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.01, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=200, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=7,
              silent=None, subsample=1, verbosity=1)
eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
G=model.fit(X_train_scaled,y_train,early_stopping_rounds=10,eval_metric=["error", "logloss","auc","map"], eval_set=eval_set, verbose=True)

# Predict labels for train set and assess accuracy
y_train_pred = G.predict(X_train_scaled)
# Predict labels for test set and assess accuracy
y_test_pred = G.predict(X_test_scaled)

#Check Accuracy Score of both training and test sets
y_test_accuracy = accuracy_score(y_test, y_test_pred)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print('Training accuracy: ', y_train_accuracy)
print('Test accuracy: ', y_test_accuracy)

#Compute F1 score due to class imbalance
from sklearn.metrics import f1_score
y_test_f1_score=f1_score(y_test, y_test_pred, average='macro')
y_train_f1_score=f1_score(y_train, y_train_pred, average='macro')

print('Training f1_score: ', y_train_f1_score)
print('Test f1_score: ', y_test_f1_score)

#Compute Cohen's Kappa score due to class imbalance
from sklearn.metrics import cohen_kappa_score
y_test_ck_score=cohen_kappa_score(y_test, y_test_pred)
y_train_ck_score=cohen_kappa_score(y_train, y_train_pred)

print('Training cohen_kappa_score: ', y_train_ck_score)
print('Test cohen_kappa_score: ', y_test_ck_score)

#################################################################################
from sklearn.metrics import auc
lr_precision, lr_recall, _ = precision_recall_curve(y_test, CB_probs)
lr_f1, lr_auc = f1_score(y_test, y_test_pred), auc(lr_recall, lr_precision)
# summarize scores
print('xGboost: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
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
#################################################################################


#Print confusion matrix for both test and train sets
cm_test = confusion_matrix(y_test, y_test_pred)
cm_train = confusion_matrix(y_train, y_train_pred)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# keep probabilities for the positive outcome only
xG_probs = model.predict_proba(X_test_scaled)
xG_probs = xG_probs[:, 1]

#Predict Probabilities- of both test and train sets
xG_train_probs = model.predict_proba(X_train_scaled)
xG_train_probs = xG_train_probs[:, 1]
xG_test_probs = model.predict_proba(X_test_scaled)
xG_test_probs = xG_test_probs[:, 1]

#Check ROC/AUC Score for test dataset
ns_auc = roc_auc_score(y_test, ns_probs)
xG_auc = roc_auc_score(y_test, xG_probs)

#testing for both test and training set
xG_test_auc = roc_auc_score(y_test, xG_test_probs)
xG_train_auc = roc_auc_score(y_train, xG_train_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('xG Boost: ROC AUC=%.3f' % (xG_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, xG_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='xG Boost')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


#Feature Importance
print(G.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()

from xgboost import plot_importance
# plot feature importance
plot_importance(G)
pyplot.show()

#############################################################################################
# retrieve performance metrics
results = m.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()
