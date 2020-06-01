# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:16:02 2020

@author: Home
"""

#Fitting a Neural network with mlrose- Hill Climb Algorithm
# split into input (X) and output (Y) variables
X = data2.iloc[:,0:5]
Y = data2.iloc[:,5]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.3, random_state = 3)


#Understanding class imbalance: 0:367,1:231
Y.value_counts()

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

##############################################################################
#Using random_hillclimb_algorithm
# Initialize neural network object and fit object
nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [5], activation = 'relu',
                                 algorithm = 'random_hill_climb', max_iters = 2000,
                                 bias = True, is_classifier = True, learning_rate = 0.001,
                                 early_stopping = True, clip_max = 5, max_attempts = 1000,
				 random_state = 3)

nn_model1.fit(X_train_scaled, y_train)


# Predict labels for train, test set and assess accuracy
y_train_pred = nn_model1.predict(X_train_scaled)
y_test_pred = nn_model1.predict(X_test_scaled)

y_train_accuracy = accuracy_score(y_train, y_train_pred)
y_test_accuracy = accuracy_score(y_test, y_test_pred)

print('Training accuracy: ', y_train_accuracy)
print('Test accuracy: ', y_test_accuracy)

#Confusion matrix
cm1 = confusion_matrix(y_test, y_test_pred)
#####################################################################################
# Initialize neural network object and fit object
nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [25], activation = 'relu',
                                 algorithm = 'gradient_descent', max_iters = 1000,
                                 bias = True, is_classifier = True, learning_rate = 0.01,
                                 early_stopping = True, clip_max = 5, max_attempts = 1000,
				 random_state = 3)


nn_model2.fit(X_train_scaled, y_train)


# Predict labels for train, test set and assess accuracy
y_train_pred = nn_model2.predict(X_train_scaled)
y_test_pred = nn_model2.predict(X_test_scaled)

y_train_accuracy = accuracy_score(y_train, y_train_pred)
y_test_accuracy = accuracy_score(y_test, y_test_pred)

print('Training accuracy: ', y_train_accuracy)
print('Test accuracy: ', y_test_accuracy)

#Confusion matrix
cm2 = confusion_matrix(y_test, y_test_pred)
cm2_1 = confusion_matrix(y_train, y_train_pred)

##############################################################################################
# Initialize logistic regression object and fit object
lr_model1 = mlrose.LogisticRegression(algorithm = 'random_hill_climb', max_iters = 1000,
                                      bias = True, learning_rate = 0.01,
                                      early_stopping = True, clip_max = 5, max_attempts = 100,
                                      random_state = 3)

lr_model1.fit(X_train_scaled, y_train)

# Predict labels for train set and assess accuracy
y_train_pred = lr_model1.predict(X_train_scaled)
# Predict labels for test set and assess accuracy
y_test_pred = lr_model1.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test, y_test_pred)
y_train_accuracy = accuracy_score(y_train, y_train_pred)


print('Training accuracy: ', y_train_accuracy)
print('Test accuracy: ', y_test_accuracy)
cm2 = confusion_matrix(y_test, y_test_pred)
cm2_1 = confusion_matrix(y_train, y_train_pred)
##############################################################################################
#Fitting Logistic Regression model-sklearn

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_scaled, y_train)
# predict probabilities
lr_probs = model.predict_proba(X_test_scaled)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
#################################################################################
#Light GBM

import lightgbm as lgb
from sklearn import metrics

def auc2(m, X_train_scaled, X_test_scaled): 
    return (metrics.roc_auc_score(y_train,m.predict(X_train_scaled)),
                            metrics.roc_auc_score(y_test,m.predict(X_test_scaled)))

lg = lgb.LGBMClassifier(silent=False)
param_dist = {"max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]
             }
grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
grid_search.fit(X_train_scaled,y_train)
grid_search.best_estimator_

d_train = lgb.Dataset(X_train_scaled, label=y_train)
params = {"max_depth": 50, "learning_rate" : 0.1, "num_leaves": 900,  "n_estimators": 300}

# Without Categorical Features
model2 = lgb.train(params, d_train)
auc2(model2, X_train_scaled, X_test_scaled)





