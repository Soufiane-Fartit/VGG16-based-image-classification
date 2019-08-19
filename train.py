import keras
import tensorflow as tf


from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import glob
import numpy as np
import cv2



dogs_train = []
cats_train = []
dogs_test = []
cats_test = []

dogs_train_target = []
cats_train_target = []
dogs_test_target = []
cats_test_target = []


for filename in glob.glob('dataset/training_set/dogs/*.jpg'): #assuming gif
  im = cv2.imread(filename)
  try:
    im = cv2.resize(im,(224,224))
    #im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    dogs_train.append(im)
    dogs_train_target.append([1,0])
  except:
    pass

for filename in glob.glob('dataset/training_set/cats/*.jpg'): #assuming gif
  im = cv2.imread(filename)
  try:
    im = cv2.resize(im,(224,224))
    #im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    cats_train.append(im)
    cats_train_target.append([0,1])
  except:
    pass

for filename in glob.glob('dataset/test_set/dogs/*.jpg'): #assuming gif
  im = cv2.imread(filename)
  try:
    im = cv2.resize(im,(224,224))
    #im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    dogs_test.append(im)
    dogs_test_target.append([1,0])
  except:
    pass

for filename in glob.glob('dataset/test_set/cats/*.jpg'): #assuming gif
  im = cv2.imread(filename)
  try:
    im = cv2.resize(im,(224,224))
    #im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    cats_test.append(im)
    cats_test_target.append([0,1])
  except:
    pass

dogs_train = np.array(dogs_train)
cats_train = np.array(cats_train)
dogs_test = np.array(dogs_test)
cats_test = np.array(cats_test)

dogs_train_target = np.array(dogs_train_target)
cats_train_target = np.array(cats_train_target)
dogs_test_target = np.array(dogs_test_target)
cats_test_target = np.array(cats_test_target)

train_set = np.concatenate((dogs_train, cats_train))
train_targets = np.concatenate((dogs_train_target, cats_train_target))

test_set = np.concatenate((dogs_test, cats_test))
test_targets = np.concatenate((dogs_test_target, cats_test_target))

train_set, train_targets = shuffle(train_set, train_targets)
test_set, test_targets = shuffle(test_set, test_targets)



config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

model = vgg2c()

for layer in model.layers[:2]:
    layer.trainable=False

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



