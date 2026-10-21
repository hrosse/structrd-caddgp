##### 
#
# Creates a local copy of converted data in numpy's npz format.
#
# Before running training and evaluation:
# Place mnist dataset files in the directory and from there run the following in command line:
# python mnist.py
#
#####

import numpy as np
import os
import gzip

os.makedirs("./mnist", exist_ok=True)

fn_xtrain = "train-images-idx3-ubyte.gz"
fn_ytrain = "train-labels-idx1-ubyte.gz"
fn_xtest = "t10k-images-idx3-ubyte.gz"
fn_ytest = "t10k-labels-idx1-ubyte.gz"

destination_xtrain = "mnist/xtrain.npz"
destination_ytrain = "mnist/ytrain.npz"
destination_xtest = "mnist/xtest.npz"
destination_ytest = "mnist/ytest.npz"

file = gzip.open(fn_xtrain,'r')
image_size = 28
num_images = 60000
file.read(16)
buf = file.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
X_train = data.reshape(num_images, image_size, image_size, 1)

file = gzip.open(fn_ytrain,'r')
num_labels = 60000
file.read(8)
buf = file.read(num_labels)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
Y_train = labels 
mnist_label_train = Y_train

file = gzip.open(fn_xtest,'r')
image_size = 28
num_images = 10000
file.read(16)
buf = file.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
X_test = data.reshape(num_images, image_size, image_size, 1)

file = gzip.open(fn_ytest,'r')
num_labels = 10000
file.read(8)
buf = file.read(num_labels)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
Y_test = labels
mnist_label_test = Y_test

mnist_images_train = X_train/255.
mnist_images_test = X_test/255.

np.savez_compressed(destination_xtrain, mnist_images_train)
np.savez_compressed(destination_ytrain, mnist_label_train)
np.savez_compressed(destination_xtest, mnist_images_test)
np.savez_compressed(destination_ytest, mnist_label_test)
