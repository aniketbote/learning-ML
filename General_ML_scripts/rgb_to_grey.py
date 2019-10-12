import numpy as np
from skimage.color import rgb2gray
import os
import cv2
from PIL import Image
from skimage import io


target_path = '../../data/chest_xray/grey'
train_path = '../../data/chest_xray/train'
classes = ['NORMAL','PNEUMONIA']
train = []
label = []

for c in classes:
    class_dir = os.path.join(train_path,c)
    count = 1
    for name in os.listdir(class_dir):
        name = os.path.join(class_dir,name)
        img = io.imread(name, as_gray=True)
        t_p = target_path + '/{}/{}.png'.format(c,count)
        cv2.imwrite(t_p,img)
        lol = cv2.imread(t_p)
        #print(lol)
        print(img)
        count=+1
        break




