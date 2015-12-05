from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

RATIO = 0.8 # The ratio of the data set to use for training
PER_CATEGORY = 98 # Images to be used per category (training + validation)
CATEGORIES = 7 # Number of categories present in the data folder
DIR = "../7Cate" # Path to folder
TYPE = ".jpg" # Extension of the images in the subfolders

DIM = 128 # Input to the network (images are resized to be square)
PREAUG_DIM = 140 # Dimensions to augment from

EPOCHS = 50
BATCH_SIZE = 1

SEED1 = 6789
SEED2 = 9876

SAVE = False


def load_dataset():


    print("Loading images")

    if len(sys.argv) == 2:
      SAVE = True
      savename = sys.argv[1]  
      print("Network parameters will be saved as " + savename + ".npy")

    folders = os.listdir(DIR)
    features = ( )

    for foldername in folders:

      if foldername.startswith("."):
        continue

      files = os.listdir(DIR +"/" + foldername)

      for file in files:
        if not file.endswith(TYPE):
          files.remove(file)

      if len(files) > PER_CATEGORY:
        files = sklearn.cross_validation.train_test_split(files, random_state=SEED1, train_size=PER_CATEGORY)[0] # discarding the "test" split

      if not len(files) == PER_CATEGORY:
        raise ValueError("Can not find " + str(PER_CATEGORY) + " images in the folder " + foldername)

      for file in files:
        img = imread(DIR +"/" + foldername + "/" + file)
        img = imresize(img, (PREAUG_DIM, PREAUG_DIM))
            #print(np.shape(img)) -> (140, 140, 3)
            #print(np.shape(features)) --> (#index, 140, 140, 3)
        features = features + (img,)

    features = np.array(list(features)) # Array conversion
    features= features.astype(theano.config.floatX) / 255.0 - 0.5

        #features = features.transpose( (0, 3, 1, 2) ) #(h, w, channel) to (channel, h, w)

        # Generate labels
    label = np.zeros(PER_CATEGORY)
    for index in range(CATEGORIES - 1):
      arr= np.full((PER_CATEGORY,), index + 1)
      label = np.append(label, arr, axis=0)

    label = label.astype("int32")

        # Split into training and validation sets
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
        features,
        label,
        random_state=SEED2,
        train_size=RATIO,
    )

    lastIndex_for_train_val = 882 * 0.6
    features_train_val = features[1:lastIndex_for_train_val,];
        #print(np.shape(features_train_val))
    labels_train_val = label[1:lastIndex_for_train_val,];
        #print(np.shape(labels_train_val))
    X_train, X_valid, y_train, y_valid = sklearn.cross_validation.train_test_split(
        features_train_val,
        labels_train_val,
        random_state=SEED2,
        train_size=0.75,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn(input_var=None):
    print("Building cnn...")

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

num_epochs=EPOCHS

print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

    # Construct the CNN
network = build_cnn(input_var)


    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
print("Starting training...")
    # We iterate over epochs:
for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True): #for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

        # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False): #for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

        # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))

    # After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))
