# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:37:18 2020

@author: Home
"""
#importing the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
from numpy import array
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
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
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.25, random_state = 3)

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Prepare a validation set-20% validation
num_val_samples = int(len(X_train_scaled) * 0.2)
#train_features = X_train_scaled[:-num_val_samples]
#train_targets = y_train[:-num_val_samples]
train_features = X_train_scaled
train_targets = y_train

#val_features = X_train_scaled[-num_val_samples:]
#val_targets = y_train[-num_val_samples:]
test_features=X_test_scaled
test_targets=y_test
print("Number of test samples:", len(test_features))
print("Number of training samples:", len(train_features))
print("Number of validation samples:", len(val_features))

#Analyze class imbalance in the targets
counts = np.bincount(train_targets)

print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(train_targets)
    )
)
print(
    "Number of negative samples in training data: {} ({:.2f}% of total)".format(
        counts[0], 100 * float(counts[0]) / len(train_targets)
    )
)
    
    
weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]

print(train_features.shape, train_features.dtype)
print(train_targets.shape, train_targets.dtype)

#Build a binary classification model

from tensorflow import keras

model = keras.Sequential(
    [
        keras.layers.Dense(
            32, activation="relu", input_shape=(6,)
        ),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()

#Train the model with class_weight argument

metrics = [
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.Accuracy(name="accuracy"),
]

model.compile(
    optimizer= 'adam', loss="binary_crossentropy", metrics=metrics
)   

callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.h5")]
class_weight = {0: weight_for_0, 1: weight_for_1} 

train_features=np.array(train_features)
train_targets=np.array(train_targets)
val_features=np.array(val_features)  
val_targets=np.array(val_targets)


history=model.fit(
    train_features,
    train_targets,
    batch_size=5,
    epochs=200 ,
    verbose=2,
    #callbacks=callbacks,
    #validation_data=(val_features, val_targets),
    class_weight=class_weight,
)  




_, _, _, _, _, _,_,  train_acc = model.evaluate(train_features, train_targets, verbose=0)
_, _, _, _, _, _,_,  val_acc = model.evaluate(val_features, val_targets, verbose=0)
print('Train: %.3f, Val: %.3f' % (train_acc, val_acc))




y_test_pred = history.predict(test_features)
y_train_pred = history.predict(train_features)
#Print confusion matrix for both test and train sets
cm_test = confusion_matrix(y_test, y_test_pred)
cm_train = confusion_matrix(y_train, y_train_pred)


#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

# plot train and validation loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

# learning curves of model accuracy
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='Val')
    pyplot.legend()
    pyplot.show()


