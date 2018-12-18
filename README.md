# ANN-CustomFalsePositiveNegativeMetrics
Write custom false positive and false negative rate metrics for back propagation

1.	Data set used is the diabetic retinopathy dataset from UCI. The dataset is attached with the homework.

2.	Keras by default uses backpropagation in its neural network. With this idea in mind, I first ran a simple neural network with the below parameters.

```
network = models.Sequential()
network.add(layers.Dense(16, input_dim=19, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))
sgd = optimizers.SGD(lr = 0.01)
network.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

The sklearn.metrics supplies the confusion_matrix function which calculates true and false negatives and positives. Using these values and the below formula, the false negative rate (FNR) and false positive rate (FPR) were calculated. The values generated were FNR = 0.256 and FPR = 0.190 with loss = 0.53 and accuracy = 0.73 at the 50th epoch.

FNR = FN/(FN+TP) 	FPR = FP/(FP+TN)

3.	I then made changes to the parameters supplied to the backpropagation model instead of making changes to the model itself. The following were the changes made:

```
network = models.Sequential()
network.add(layers.Dense(32, input_dim=19, activation='relu'))
network.add(layers.Dense(16, input_dim=32, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))
sgd = optimizers.SGD(lr = 0.10)
network.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])`
```

The threshold values for the model were provided during compile time to custom metric functions which calculate FPR and FNR for every epoch. The code is given below:

```
import keras.backend as K

def falsePositiveRate(threshold=0.5):
    def FPR(y_true, y_pred):
        threshold_value = threshold
        y_pred = K.cast(K.less(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        true_negatives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        predicted_negatives = K.sum(y_pred)
        precision_ratio = true_negatives / (predicted_negatives + K.epsilon())
        return ((1-precision_ratio)*100)
    return FPR

def falseNegativeRate(threshold = 0.5):
    def FNR(y_true, y_pred):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return ((1-recall_ratio)*100)
    return FNR
```

`network.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy',falsePositiveRate(0.6),falseNegativeRate(0.4)])`

Note: These custom metrics could not be passed as custom loss function because the loss function requires tensor objects which are differentiable to calculate the gradient descent. The FPR and FNR values always have to use the comparison operator to calculate its values and these comparison operator tensor objects cannot be differentiated.

For the different threshold values passed, the FPR, FNR, accuracy and loss metrics are listed in the table below:

```
Threshold
(FPR, FNR)	  FPR	    FNR	      Accuracy	    Loss
0.5, 0.5	  80%	    20%	        0.84	    0.35
0.4, 0.5	  84.58%    20%	        0.82	    0.34
0.6, 0.4	  77.88%    11.06%      0.84	    0.34
0.7, 0.3	  74.3%     8.2%	0.84	    0.34
0.8, 0.2	  69.47%    5.02%	0.84	    0.34
0.9, 0.1	  65.71%    0.85%	0.856	    0.336
```

4.	The observations made were that accuracy and loss remains pretty much the same after a while. The minimum observable FPR achievable for threshold above 0.9 is 65.71% and FNR for threshold below 0.1 is 0.85%. These values are for a fixed learning rate.
