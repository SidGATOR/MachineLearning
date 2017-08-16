import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def sigmoid(x):

    return 1/(1 + np.exp(-x))


raw_data  = pd.read_csv("binary.csv")
target = raw_data['admit']

## Last error
last_loss = None

## Preprocessing Data
# One-Hot encoding of catagorical data
data = raw_data.drop('admit',axis=1)
data = pd.get_dummies(data, columns=['rank'])
# Standardization of GRE and Grade scores
# make the distribution of these columns with 0 mean and std of 1
for field in ['gre','gpa']:
    mean,std = np.mean(data[field]) , np.std(data[field])
    data.loc[:,field] = (data[field] - mean) / std

## Drawing weights from normal distribution
n_records, n_features = data.shape
np.random.seed(42)
weights = np.random.normal(scale= 1/n_features**.5,size=n_features)

## Spliting data for testing and training
train_features, test_features, train_labels, test_labels = train_test_split(data,target,test_size=0.10,random_state=42)

## Neural Network Hyperparamters
epochs = 1000
learning_rate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(np.array(train_features),np.array(train_labels)):

        output = np.dot(x,weights)
#        print ("output: {:}, x: {:}, y: {:}" .format(output,x,y))
        activation = sigmoid(output)
#        print ("Actication out: {:}, output_sum: {:}" .format(activation,output))

        error = y - activation

        sigmoid_grad = activation * (1 - activation )

        error_term = error * sigmoid_grad

       # print ("error_term: {:}, x: {:}".format(error_term,x))
        del_w += learning_rate * error_term * x

    weights += del_w

    if e % (epochs / 10) == 0:
        sig_out = sigmoid(np.dot(np.array(train_features),weights))
        loss = np.mean((np.array(train_labels) - sig_out) ** 2)
        if last_loss and last_loss < loss:
            print ("Training loss: {:}, WARNING - Loss increaseing" .format(loss))
        else:
            print ("Training loss: {:}" .format(loss))


## Tesing on test data ##
tes_out = sigmoid(np.dot(np.array(test_features),weights))
predictions = tes_out > 0.5
accuracy = np.mean((np.array(test_labels) == predictions))
print ("Prediction Accuracy: {}" .format(accuracy))

