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

## Spliting data for testing and training
train_features, test_features, train_labels, test_labels = train_test_split(data,target,test_size=0.10,random_state=42)

# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = train_features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(np.array(train_features), np.array(train_labels)):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_input = np.dot(x,weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(np.dot(hidden_output,weights_hidden_output))

        ## Backward pass ##
        # TODO: Calculate the network's prediction error
        error = y - output

        # TODO: Calculate error term for the output unit
        output_error_term = error * output * ( 1 - output )

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term,weights_input_hidden)

        # TODO: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        # TODO: Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x[:,None]

    # TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - np.array(train_labels)) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(np.array(test_features), weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == np.array(test_labels))
print("Prediction accuracy: {:.3f}".format(accuracy))

