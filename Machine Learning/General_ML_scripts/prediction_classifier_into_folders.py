from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import os
import pickle
import numpy as np
import shutil

src_pne = 'pne'
src_nonpne = 'nonpne'
os.mkdir(src_pne)
os.mkdir(src_nonpne)
filename = 'final1.sav' #path to the saved model
loaded_model = pickle.load(open(filename, 'rb'))

path_test='report' # path to the image folder
for names in sorted(os.listdir(path_test)):  
    img = image.load_img(os.path.join(path_test,names), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = loaded_model.predict(x)
    y_classes = preds.argmax(axis=-1)
    print(y_classes[0])
    if y_classes[0] == 1:
        shutil.copy(os.path.join(path_test, names), src_pne)
    else:
        shutil.copy(os.path.join(path_test, names),src_nonpne)
     #predictions
