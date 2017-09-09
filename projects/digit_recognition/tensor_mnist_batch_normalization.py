import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tqdm
#import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets('.',one_hot=True,reshape=False)

# Save File #
save_file = "./models/save_test_model"

## Input Parameters ##
n_classes = 10
last_loss = None

## Model Hyperparameters ##
epochs = 500
learning_rate = 0.0001
dropout = 0.5
test_size = 256
batch_size = 50
epsilon = 1e-3
decay = 0.999

## Definition ##

def neural_net_image_input(image_shape):

    image_shape = [None,image_shape[0],image_shape[1],image_shape[2]]
    X = tf.placeholder(tf.float32,shape=image_shape,name="x")
    return X


def neural_net_label_input(n_classes):

    Y = tf.placeholder(tf.float32,[None,n_classes],name="y")
    return Y


def neural_net_keep_prob_input():

    keep_prob = tf.placeholder(tf.float32,name="keep_prob")
    return keep_prob

def batch_norm_wrapper(x_tensor,is_training):

    ## Tensor Shape ##
    tensor_shape = x_tensor.get_shape().as_list()
    ## Gamma and Beta ##
    gamma = tf.Variable(tf.ones(shape=[tensor_shape[-1]]))
    beta = tf.Variable(tf.zeros(shape=[tensor_shape[-1]]))
    pop_mean = tf.Variable(tf.zeros(shape=[tensor_shape[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones(shape=[tensor_shape[-1]]), trainable=False)

    if is_training:
        if len(tensor_shape) == 2:
            batch_mean, batch_var = tf.nn.moments(x_tensor,axes=[0])
        elif len(tensor_shape) == 4:
            batch_mean, batch_var = tf.nn.moments(x_tensor,axes=[0,1,2])
        else:
            print("Wrong Dimensions")
            exit()

        train_mean = tf.assign(pop_mean,pop_mean * decay + (1 - decay) * batch_mean)
        train_var = tf.assign(pop_var,pop_var * decay + (1 - decay) * batch_var)

        with tf.control_dependencies([train_mean,train_var]):
            return tf.nn.batch_normalization(x_tensor,batch_mean,batch_var,beta,gamma,epsilon)
    else:
        return tf.nn.batch_normalization(x_tensor,pop_mean,pop_var,beta,gamma,epsilon)


def xavier_init(x_tensor,weight_dim):

    Nin = x_tensor.get_shape().as_list()[-1]
    Nout = weight_dim[-1]

    return tf.cast(tf.sqrt(tf.divide(2,tf.add(Nin,Nout))),tf.float32)


def conv2d_maxpool(x_tensor,conv_num_outputs,conv_ksize,conv_stride,pool_ksize,pool_stride,is_training):

    ## Tensor Shape ##
    # [batch,height,width,depth]
    tensor_shape = x_tensor.get_shape().as_list()

    ## Weight and Bias Dimensions #
    weight_dim = [*conv_ksize,tensor_shape[-1],conv_num_outputs]
    bias_dim = [conv_num_outputs]

    ## Filter Dimensions ##
    filter_stride = [1,*conv_stride,1]

    ## Pooling Dimensions ##
    pool_stride = [1,*pool_stride,1]
    pool_ksize = [1,*pool_ksize,1]

    ## Weights and Biases ##
    weights = tf.Variable(tf.truncated_normal(shape=weight_dim,stddev=xavier_init(x_tensor,weight_dim)))
    biases = tf.Variable(tf.constant(0.1,shape=bias_dim))

    ## Convolution ##
    conv_layer = tf.nn.bias_add(tf.nn.conv2d(x_tensor,weights,filter_stride,padding="SAME"),biases)

    ## Batch Normalization ##
    conv_layer = batch_norm_wrapper(conv_layer,is_training)

    ## Activation ##
    conv_layer = tf.nn.relu(conv_layer)

    ## Max Pooling ##
    conv_layer = tf.nn.max_pool(conv_layer,pool_ksize,pool_stride,padding="SAME")

    return conv_layer


## Change tensor from 4D to 2D for dense layers
def flatten(x_tensor):

    ## Tensor shape ##
    tensor_shape = x_tensor.get_shape().as_list()
    tensor_shape = tensor_shape[1] * tensor_shape[2] * tensor_shape[3]

    return tf.reshape(x_tensor,shape=[-1,tensor_shape])

def fully_conn(x_tensor,num_outputs,is_training):

    ## Tensor shape ##
    tensor_shape = x_tensor.get_shape().as_list()

    ## Weight and Bias Dimensions ##
    weight_dim = [tensor_shape[-1],num_outputs]
    bias_dim = [num_outputs]

    ##Weights and Biases
    weights = tf.Variable(tf.truncated_normal(shape=weight_dim,stddev=xavier_init(x_tensor,weight_dim)))
    biases = tf.Variable(tf.constant(0.1,shape=bias_dim))

    ## Forward Propogation ##
    fc1 = tf.add(tf.matmul(x_tensor,weights),biases)

    ## Batch Normalization ##
    fc1 = batch_norm_wrapper(x_tensor,is_training)

    ## Activation ##
    fc1 = tf.nn.relu(fc1)

    return fc1


def output(x_tensor,num_outputs):

    ## Weight and Bias Dimensions ##
    weight_dim = [x_tensor.get_shape().as_list()[-1],num_outputs]
    bias_dim = [num_outputs]

    ## Weights and Biases ##
    weights = tf.Variable(tf.truncated_normal(shape=weight_dim,stddev=xavier_init(x_tensor,weight_dim)))
    biases = tf.Variable(tf.constant(0.1,shape=bias_dim))

    ## Forward Propogation ##
    out = tf.add(tf.matmul(x_tensor,weights),biases,name="output")

    return out

def conv_net(is_training,keep_prob):

    ## Features and Labels ##
    features = neural_net_image_input((28,28,1))
    labels = neural_net_label_input((10))


    ## Convolutions layer parameters ##
    conv_param = {"conv1_num_outputs" : 32, "conv1_conv_ksize" : (5,5), "conv1_conv_strides" : (1,1),
                  "conv1_pool_ksize" : (2,2), "conv1_pool_strides" : (2,2), "conv2_num_outputs" : 64,
                  "conv2_conv_ksize" : (5,5), "conv2_conv_strides" : (1,1), "conv2_pool_ksize" : (2,2),
                  "conv2_pool_strides" : (2,2), "conv3_num_outputs" : 128 , "conv3_conv_ksize" : (3,3),
                  "conv3_conv_strides" : (1,1), "conv3_pool_ksize" : (2,2),"conv3_pool_strides" : (2,2) }

    ## Parameters: Fully Connected Layer ##
    fc_param = {"fc1_num_outputs" : 1024 , "fc2_num_outputs" : 256,"fc3_num_outputs" : 64, "dropout": keep_prob }

    ## Paramaters: Output layer ##
    output_param = {"output_num_outputs" : 10}


    ## Layer 1 ##
    conv1_layer = conv2d_maxpool(features,conv_param["conv1_num_outputs"], conv_param["conv1_conv_ksize"], conv_param["conv1_conv_strides"],
                                 conv_param["conv1_pool_ksize"],conv_param["conv1_pool_strides"],is_training)

    ## Layer 2 ##
    conv2_layer = conv2d_maxpool(conv1_layer,conv_param["conv2_num_outputs"], conv_param["conv2_conv_ksize"], conv_param["conv2_conv_strides"],
                                 conv_param["conv2_pool_ksize"],conv_param["conv2_pool_strides"],is_training)

    ## Flattening ##
    # Convert from 4D to 2D
    flat = flatten(conv2_layer)

    ## Fully Connected Layer 1 ##
    fc1_layer = fully_conn(flat,fc_param["fc1_num_outputs"],is_training)
    # Dropout #
    fc1_layer = tf.nn.dropout(fc1_layer,keep_prob=keep_prob)
    """
    ## Fully Connected Layer 2 ##
    fc2_layer = fully_conn(fc1_layer,fc_param["fc1_num_outputs"],is_training)
    # Dropout #
    fc2_layer = tf.nn.dropout(fc2_layer,keep_prob=keep_prob)
    """
    ## Output Layer ##
    out_layer = output(fc1_layer,output_param["output_num_outputs"])

    ## Cost or Cross Entropy Loss ##
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=labels))

    ## Optimizer ##
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    ## Predictions ##
    predictions = tf.equal(tf.argmax(out_layer,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(predictions,tf.float32), name="accuracy")

    return features, labels, cost, optimizer, accuracy, tf.train.Saver()

## Dropout Probability ##
keep_probability = neural_net_keep_prob_input()
features, labels, cost, optimizer, accuracy, saver = conv_net(True,keep_probability)

acc = []

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    total_batches = int(mnist.train.num_examples / batch_size)

    for epoch in tqdm.tqdm(range(epochs)):

        #for i in range(total_batches):

        batch_x, batch_y = mnist.train.next_batch(batch_size)
        ## Run Optimizer ##
        sess.run(optimizer,feed_dict={features : batch_x, labels : batch_y, keep_probability : dropout})

        if epoch % (epochs / 10) == 0:
            loss = sess.run(cost,feed_dict={features : batch_x, labels : batch_y, keep_probability : 1.0})
            vali_acc = sess.run(accuracy, feed_dict={features : mnist.validation.images, labels: mnist.validation.labels, keep_probability : 1.0})
            #acc.append(vali_acc)
            if last_loss and last_loss > loss:
                saver.save(sess,save_file)
            else:
                print("Validation loss has not decreased")
            last_loss = loss
            print("Epoch #: {:}, Loss: {:}, Validation Accuracy: {:} " .format(epoch+1,loss,vali_acc))

    test_acc = sess.run(accuracy,feed_dict={features : mnist.test.images, labels : mnist.test.labels, keep_probability : 1.0})
    print("Validation Accuracy: {:} " .format(test_acc))
"""
tf.reset_default_graph()
keep_probability = neural_net_keep_prob_input()
feautres, labels, cost, optimizer, accuracy,loader = conv_net(False,keep_probability)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loader.restore(sess,save_file)
    test_acc = sess.run([accuracy,keep_probability],feed_dict={features : mnist.test.images[:test_size], labels: mnist.test.labels[:test_size], keep_probability : 1.0})
    print("Test Accuracy: {:}, Keep_prob: {:} " .format(test_acc[0], test_acc[1]))

"""
"""
acc = np.array(acc)

fig, ax = plt.subplots()

ax.plot(range(0,len(acc)*100,100),acc,label="With BN")
#ax.plot(range(0,len(acc_bn)*100,100), acc_bn,label="With BN")
ax.set_xlabel("Training Epochs")
ax.set_ylabel("Accuracy")
ax.set_title("Batch Normalization Validation Accuracy")
ax.legend(loc=4)
plt.show()
"""

