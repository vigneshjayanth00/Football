# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:39:53 2020

@author: VigneshJayanth
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
X = data2.iloc[:,0:6]
Y = data2.iloc[:,6]


# Split data into training and test sets
# fit and apply the transform
# define undersampling strategy
undersample = RandomUnderSampler(sampling_strategy='majority')

X_train_under, y_train_under = undersample.fit_resample(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X_train_under, y_train_under,
                                                    test_size = 0.35, random_state = 3)


# Step 2: Make an instance of the Model
tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=5)
clf.fit(X_train_under, y_train_under)
clf_under = DecisionTreeClassifier(criterion='entropy', max_depth=5)

#Undersampling results
clf_under.fit(X_train, y_train)

y_pred = clf_under.predict(X_test)
y_train_pred = clf_under.predict(X_train)

#Accuracy,f1_score,recall_score
rs_under=recall_score(y_test, y_pred)
accu_under=accuracy_score(y_test, y_pred)
f1_under=f1_score(y_test, y_pred)

print('F1 Score: %.3f' % f1_under)
print('Accuracy Score: %.3f' % accu_under)
print('Recall Score: %.3f' % rs_under)

#Print confusion matrix for both test and train sets
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_train_pred)

# predict probabilities
DT_probs = clf.predict_proba(X_test)
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# keep probabilities for the positive outcome only
DT_probs = DT_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
dt_auc = roc_auc_score(y_test, DT_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, DT_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Decision-Tree')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()



from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))