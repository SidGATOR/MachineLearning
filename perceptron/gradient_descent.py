import numpy as np

## Activation Function : Sigmoid ##
def sigmoid(x):

    return 1/(1 + np.exp(-x))

## 1st Derivative of Activation Function ##

def sigmoid_prime(x):

    return (sigmoid(x) * (1- sigmoid(x)))

## Input Data
x = np.array([1,2,3,4])

## Target Value
y = 0.5

## Input Weights
weights = np.array([0.5,-0.5,0.3,0.1])

## Learning Rate
learnRate = 0.5

## Forward Propogation Run ##
output_sum = np.dot(x,weights)

# Sigmoid output #
sigmoid_out = sigmoid(output_sum)

## Error in output ##
error = y - output_sum

## Gradient of output
#output_grad = sigmoid_prime(output_sum)
# Efficiency tip #
output_grad = sigmoid_out * ( 1 - sigmoid_out )

error_term = error * output_grad

del_w = learnRate * error_term * x



print ("Neural Network output:")
print (output_sum)
print ("Amount of error: {:}" .format(error))
print ("Change in weights: {:}" .format(del_w))




