import os
import numpy as np
import tensorflow as tf

np.random.seed(101)

destination_xtrain = "xtrain.npz"
destination_ytrain = "ytrain.npz"
destination_xtest = "xtest.npz"
destination_ytest = "ytest.npz"

fpath = "planesnet/"
img_list = []
label_list = []
fname_list = []
for im_path in os.listdir(fpath):
    fname_list.append(im_path)
    label_list.append(int(im_path[0]))
    image = tf.io.read_file(fpath + im_path)
    image = np.divide(tf.io.decode_png(image).numpy(), 255, dtype=np.float32)
    img_list.append(image)


images = np.array(img_list)
labels = np.array(label_list)

testind = np.random.choice(range(images.shape[0]), size=int(images.shape[0]/4), replace=False) # 25 percent test data, rest train
mask = np.ones(images.shape[0], dtype=bool)
mask[testind] = False

X_train = images[mask]
Y_train = labels[mask]

X_test = images[mask == False]
Y_test = labels[mask == False]

np.savez_compressed(destination_xtrain, X_train)
np.savez_compressed(destination_ytrain, Y_train)
np.savez_compressed(destination_xtest, X_test)
np.savez_compressed(destination_ytest, Y_test)