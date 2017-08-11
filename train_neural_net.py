
import tensorflow as tf
import numpy as np
import os
from os import path
import csv

# constants
INPUT_SIZE = 8
HIDDEN_SIZE = 20
OUTPUT_SIZE = 1

LEARNING_RATE = 1

# placeholders
X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="X")
y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="y")

def normalize(X):
    """
    Normalizes each column of the feature matrix X (with rows as feature vectors), and returns the new feature matrix X, and the mu and sigma used. (mu = mean, sigma = standard deviation).
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    normX = (X - mu) / sigma

    return normX, mu, sigma


def normalizeWith(X, mu, sigma):
    """
    Performs normalization on each column of the feature vector X using the mu and sigma provided.
    """
    normX = (X - mu) / sigma
    return normX


def layer(inp, input_units, output_units, layer_num):
    """
    Creates a fully-connected neural network layer, given the number of input and output units, and the input.

    Returns the layer created, and the weight corresponding to it (for regularization)
    """
    Weights = tf.Variable(tf.random_normal([input_units, output_units]), name="weight" + str(layer_num))

    biases = tf.Variable(tf.random_normal([output_units]), name="bias" + str(layer_num))

    # since this is vectorized, the inp will be rows of training examples, so we think of the inputs as a row vector instead of a column vector (usual NN model).
    layer = tf.matmul(inp, Weights) + biases

    layer = tf.sigmoid(layer)

    return layer, Weights


def feedforward(X, BETA):
    """
    Implements feed-forward propagation using tensorflow, and uses the provided inputs (X), weights and biases.

    Args:
        X - a matrix containing rows of training examples.
        weights - a dictionary containing Tensorflow variables, all of which are matrices with the corresponding dimensions of the layer they belong to.
        biases - a dictionary containing Tensorflow variables, which are column vectors containing the bias units.

    Returns:
        - The final layer of the network, containing the predictions (the hypothesis)
        - The cost function implementing regularization with provided BETA
    """

    # making first layer - think of it as the hidden layer
    layer_1, weight_1 = layer(X, INPUT_SIZE, HIDDEN_SIZE, 1)

    # second hidden layer
    layer_2, weight_2 = layer(layer_1, HIDDEN_SIZE, HIDDEN_SIZE, 2)

    # creating the last layer, with the buy/sell prediction
    output, weight_3 = layer(layer_2, HIDDEN_SIZE, OUTPUT_SIZE, 3)

    cost_func = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))

    cost_func = tf.reduce_mean(cost_func + BETA * (tf.nn.l2_loss(weight_1) + tf.nn.l2_loss(weight_2) + tf.nn.l2_loss(weight_3)))


    return output, cost_func

# TODO : make this method use something appropriate for large data files instead of loading everything into memory.

def load_data(stock, split=(0.6, 0.2, 0.2)):
    """
    Loads the required data into six numpy matrices, returned as a dictionary - the training set (X_train and y_train), the cross-validation set (X_val and y_val) and the test set (X_test and y_test). The optional stock parameter dictates the stock ticker to return - if it is left blank, all stocks are combined into one large dataset. The optional split parameter dictates the training/validation/test split.

    By default, the first column of the data (the date) is taken off, as it is not required for learning.
    """

    data = {}

    raw_data = np.loadtxt("data/table_" + stock + ".csv", delimiter=',')

    trainingExamples = raw_data.shape[0]

    # make order random and cast to proper float32 value so that tensorflow doesn't complain
    np.random.shuffle(raw_data)
    raw_data = raw_data.astype(np.float32)

    # get all X data, skipping the date (first column) and the y labels (last column)
    X = raw_data[:, 1:-1]
    y = raw_data[:, -1].reshape(trainingExamples, 1) # ensure that we get two indices for matrix multiplication (else tensorflow complains because of the numpy array representation)

    first_split = int(round(trainingExamples * split[0]))
    second_split = first_split + int(round(trainingExamples * split[1]))

    data["X_train"] = X[0:first_split, :]
    data["y_train"] = y[0:first_split, :]

    data["X_val"] = X[first_split + 1:second_split, :]
    data["y_val"] = y[first_split + 1:second_split, :]

    data["X_test"] = X[second_split + 1:, :]
    data["y_test"] = y[second_split + 1:, :]


    #normalizing X_train
    data["X_train"], mu, sigma = normalize(data["X_train"])
    data["X_val"] = normalizeWith(data["X_val"], mu, sigma)
    data["X_test"] = normalizeWith(data["X_test"], mu, sigma)

    return data



prediction, cost_func = feedforward(X, 1)


# optimizer to use and training step computation
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
step = optimizer.minimize(cost_func)

init = tf.global_variables_initializer()



with tf.Session() as sess:
    # initialize variables
    sess.run(init)

    # loading training data
    stock = load_data("msft")

    for i in range(10000):

        _, cost = sess.run([step, cost_func], feed_dict={X : stock["X_train"], y: stock["y_train"]})

        # gradually print the current cost
        if i % 1000 == 0:
            print "Cost:", cost

    # create vector with the predictions that the classifier got correct.
    correct_predictions = tf.equal(tf.round(prediction), y)

    # graph that prints out the accuracy of the model, using the variables that have been devised by backprop
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # print accuracies with different training sets.
    print "Accuracy on training set: {acc}%\n".format(acc= sess.run(accuracy, feed_dict={X: stock["X_train"], y: stock["y_train"]}) * 100)
    print "Accuracy on validation set: {acc}%\n".format(acc= sess.run(accuracy, feed_dict={X: stock["X_val"], y: stock["y_val"]}) * 100)
    print "Accuracy on test set: {acc}%\n".format(acc= sess.run(accuracy, feed_dict={X: stock["X_test"], y: stock["y_test"]}) * 100)
