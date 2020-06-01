#CATboost

import catboost as cb
cat_features_index = [0,1,2,3,4,5,6]

def auc(m, train, test): 
    return (metrics.roc_auc_score(train,m.predict_proba(train)[:,1]),
                            metrics.roc_auc_score(test,m.predict_proba(test)[:,1]))

params = {'depth': [4, 7, 10],
          'learning_rate' : [0.03, 0.1, 0.15],
         'l2_leaf_reg': [1,4,9],
         'iterations': [10]}
cb = cb.CatBoostClassifier()
grid_search = GridSearchCV(cb, params, scoring="roc_auc", cv = 3)
grid_search.fit(X_train,y_train)

#Fit the params
clf = cb.CatBoostClassifier(eval_metric="AUC", depth=8, iterations= 250, l2_leaf_reg= 9, learning_rate= 0.001)
clf.fit(X_train,y_train)
# Predict labels for train set and assess accuracy
y_train_pred = clf.predict(X_train)
# Predict labels for test set and assess accuracy
y_test_pred = clf.predict(X_test)

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
CB_probs = clf.predict_proba(X_test)
CB_probs = CB_probs[:, 1]

#Predict Probabilities- of both test and train sets
CB_train_probs = clf.predict_proba(X_train)
CB_train_probs = CB_train_probs[:, 1]
CB_test_probs = clf.predict_proba(X_test)
CB_test_probs = CB_test_probs[:, 1]

#Check ROC/AUC Score for test dataset
ns_auc = roc_auc_score(y_test, ns_probs)
CB_auc = roc_auc_score(y_test, CB_probs)

#testing for both test and training set
CB_test_auc = roc_auc_score(y_test, CB_test_probs)
CB_train_auc = roc_auc_score(y_train, CB_train_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Cat Boost: ROC AUC=%.3f' % (CB_auc))
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
# plot
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')


#################################################################################
from sklearn.metrics import auc
lr_precision, lr_recall, _ = precision_recall_curve(y_test, CB_probs)
lr_f1, lr_auc = f1_score(y_test, y_test_pred), auc(lr_recall, lr_precision)
# summarize scores
print('Catboost: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Catboost')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
#################################################################################


