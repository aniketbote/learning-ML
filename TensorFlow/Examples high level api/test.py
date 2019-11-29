import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from keras.models import load_model
from sklearn.metrics import accuracy_score
from skimage import io

test_path = '../../data/chest_xray/test'
classes = ['NORMAL','PNEUMONIA']
train = []
label = []

#dataset loading
for c in classes:
    class_dir = os.path.join(test_path,c)
    for name in os.listdir(class_dir):
        name = os.path.join(class_dir,name)
        #im = cv2.imread(name)
        im = io.imread(name, as_gray=True)
        print(im.shape)
        im = cv2.resize(im,(40,40))
        train.append(im)
        if c == 'PNEUMONIA':
            label.append(1)
        elif c == 'NORMAL':
            label.append(0)
            

            
train = np.asarray(train)
label = np.asarray(label)
X,y = shuffle(train,label,random_state = 1)

np.save('X_t',X)
np.save('y_t',y)


X = np.load('X_t.npy')
y = np.load('y_t.npy')


X = X / 255.0

model = load_model('partly_trained_1.h5')

predictions = model.predict(X)
l = []
for i in predictions:
    p = np.argmax(i)
    l.append(p)

l = np.asarray(l)

print(l)
##score = accuracy_score(y,l)
##print(score)
##
##test_loss, test_acc = model.evaluate(X, y)
##
##print(test_acc)
