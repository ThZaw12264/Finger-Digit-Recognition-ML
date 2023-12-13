import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import imageio.v3 as iio

from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss

from sklearn.linear_model import LogisticRegression

NUM_IMAGES = 12007
IMAGE_SIZE = 1024


images = np.zeros(shape=(NUM_IMAGES,IMAGE_SIZE))
images_y = np.zeros(shape=(NUM_IMAGES,1))

# Process images into numpy arrays
for i, im_path in enumerate(glob.glob("../training_images/*.png")):
     im = iio.imread(im_path).reshape(IMAGE_SIZE,)
     target = im_path[-5]
     images[i], images_y[i]= im, target


seed = 100101

# Split into training and test data
im_train, im_test, y_train, y_test = train_test_split(images, images_y, test_size=0.1, shuffle=False, random_state=seed)

# im_tr_subset = im_train[:1000]
# y_tr_subset = y_train[:1000]

# Fit training data to learner and predict
learner = LogisticRegression(random_state=seed, max_iter=1000).fit(im_train, y_train.ravel())
pred_train = learner.predict(im_train)
pred_test = learner.predict(im_test)
print(f'Training Error Rate: {zero_one_loss(pred_train, y_train)}')
print(f'Test Error Rate: {zero_one_loss(pred_test, y_test)}')


# Heatmap of coefficients for each class
fig, ax = plt.subplots(1,6, figsize=(18,8))
mu = learner.coef_.mean(0).reshape(32,32)
for i in range(6):
    ax[i].imshow(learner.coef_[i,:].reshape(32,32)-mu,cmap='seismic',vmin=-1/100,vmax=1/100); 
    ax[i].set_title(f'Class {i-1}')
    ax[i].axis('off')
plt.show()


# Get wrong predictions and produce their images
wrong = []
for i in range(pred_test.size):
     if pred_test[i] != y_test[i]:
          wrong.append(i)

fig, ax = plt.subplots(10,4, figsize=(32,2))
fig.subplots_adjust(hspace=1)
for i, ind in enumerate(wrong):
     j = i // 4
     k = i % 4
     ax[j,k].imshow(im_test[ind].reshape(32,32), cmap ="gray", vmin=0, vmax=255)
     ax[j,k].set_title(f'Predicted {int(pred_test[ind])}', size=12)
     ax[j,k].axis('off')
plt.show()
