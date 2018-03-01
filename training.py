import numpy as np
import tensorflow as tf
import cv2
import sys
from tensorflow.examples.tutorials.mnist import input_data

# Getting input
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# TODO

# Weights and biases
def create_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
	return tf.Variable(tf.constant(0.05, shape=[size]))

# Creating network layers
def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
	# define the weights that will be trained using create_Weights function
	weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
	print("Weights: " + str(weights))

	# create biases using the function. These are also trained
	biases = create_biases(num_filters)
	print("Biases: " + str(biases))

	# Creating the convolutional layer
	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')

	# What does this mean?
	layer += biases

	# Max pool layer
	layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	# feed output into Relu which is our activation
	layer = tf.nn.relu(layer)

	return layer

# Convert the multidimensional tensor to a flat one dim tensor. Reshape does this
def create_flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer = tf.reshape(layer, [-1, num_features])

	# How does this last one function?
	# TODO: learn tensor reshaping
	return layer

# define a fully connected layer
def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
	# Define trainable weights and biases
	# I still don't understand where these weights are coming from. Arbitrary starting weights?
	weights = create_weights(shape=[num_inputs, num_outputs])
	biases = create_biases(num_outputs)

	layer = tf.matmul(input, weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)

	return layer


# Now for the actual thing
if __name__ == '__main__':

	classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	# input images here

	# get something from input data

	# placeholders for input
	x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')


	num_classes = len(classes)
	y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
	y_true_cls = tf.argmax(y_true, dimensions=1)

	layer_conv1 = create_convolutional_layer(input=x, num_input_channels=num_channels, conv_filter_size=filter_size_conv1)
	layer_conv2 = create_convolutional_layer(input=layer_conv1, num_input_channels=num_filters_conv1