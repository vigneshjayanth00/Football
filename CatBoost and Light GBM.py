# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:26:09 2020

@author: Home
"""


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

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
grid_search.fit(X_train_scaled,y_train)
#####After this step: type grid_search.best_estimator_ in the ipython console for the best estimator
d_train = lgb.Dataset(X_train_scaled, label=y_train)
params = {"max_depth": 25, "learning_rate" : 0.01, "num_leaves": 300,  "n_estimators": 200}

# Without Categorical Features
model2 = lgb.train(params, d_train)
auc2(model2, X_train_scaled,y_train)


#CATboost

import catboost as cb1
cat_features_index = [0,1,2,3,4,5,6]

def auc(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
                            metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))

params = {'depth': [4, 7, 10],
          'learning_rate' : [0.03, 0.1, 0.15],
         'l2_leaf_reg': [1,4,9],
         'iterations': [300]}
cb = cb.CatBoostClassifier()
cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv = 3)
cb_model.fit(X_train_scaled,y_train)

#Fit the params
clf = cb1.CatBoostClassifier(eval_metric="AUC", depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
clf.fit(X_train_scaled,y_train)
auc(clf, X_train_scaled,y_train)
# Predict labels for train set and assess accuracy
y_train_pred = clf.predict(X_train_scaled)
# Predict labels for test set and assess accuracy
y_test_pred = clf.predict(X_test_scaled)

#Check Accuracy Score of both training and test sets
y_test_accuracy = accuracy_score(y_test, y_test_pred)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print('Training accuracy: ', y_train_accuracy)
print('Test accuracy: ', y_test_accuracy)

#Print confusion matrix for both test and train sets
cm_test = confusion_matrix(y_test, y_test_pred)
cm_train = confusion_matrix(y_train, y_train_pred)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# keep probabilities for the positive outcome only
CB_probs = model.predict_proba(X_test_scaled)
CB_probs = CB_probs[:, 1]

#Predict Probabilities- of both test and train sets
CB_train_probs = model.predict_proba(X_train_scaled)
CB_train_probs = CB_train_probs[:, 1]
CB_test_probs = model.predict_proba(X_test_scaled)
CB_test_probs = CB_test_probs[:, 1]

#Check ROC/AUC Score for test dataset
ns_auc = roc_auc_score(y_test, ns_probs)
CB_auc = roc_auc_score(y_test, CB_probs)

#testing for both test and training set
CB_test_auc = roc_auc_score(y_test, CB_test_probs)
CB_train_auc = roc_auc_score(y_train, CB_train_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('xG Boost: ROC AUC=%.3f' % (CB_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, CB_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='CatBoost')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

#Feature Importance
print(clf.feature_importances_)
# plot
pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
pyplot.show()



