import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('.',one_hot=True,reshape=False)


# load File #
load_meta =  "./models/save_test_model.meta"
load_file = "./models/save_test_model"
test_size = 256

with tf.Session() as sess:

	## Loading the meta graph first. Remember, it will append the graph from the saved file to the existing graph
	loader = tf.train.import_meta_graph(load_meta)
	## Restoring the Variables of the Graph 
	loader.restore(sess, load_file)
	### Restoring tensors by name ###
	features = sess.graph.get_tensor_by_name("x:0")
	labels = sess.graph.get_tensor_by_name("y:0")
	keep_probability = sess.graph.get_tensor_by_name("keep_prob:0")
	accuracy = sess.graph.get_tensor_by_name("accuracy:0")

	test_acc = sess.run([accuracy,keep_probability],feed_dict={features : mnist.test.images[:test_size], labels: mnist.test.labels[:test_size], keep_probability : 1.0})
	print("Test Accuracy: {:}, Keep_prob: {:} " .format(test_acc[0], test_acc[1]))
