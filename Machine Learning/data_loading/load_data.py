import numpy as np
import pandas as pd
from skimage import io
import cv2
import os
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

def csv_to_numpy(fname, end_col, target_col = None, start_col = 0) :
    columns = list(range(start_col,end_col))
    data = np.genfromtxt(fname, max_rows = 20000, delimiter=',', skip_header = 1, usecols = columns)
    if target_col == None:
        return data
    else:
        labels = np.genfromtxt(fname, max_rows = 20000, delimiter=',', skip_header = 1, usecols=(target_col))
        return data,labels
def csv_to_dataframe(path):
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    return df

def imagedir_to_numpy(train_path,save = False):
    train = []
    label = []
    label1 = []
    classes = list(os.listdir(train_path))
    count = 0

    for c in classes:
        class_dir = os.path.join(train_path,c)
        for name in os.listdir(class_dir):
            name = os.path.join(class_dir,name)
            #im = cv2.imread(name)             #Use for RGB images
            im = io.imread(name, as_gray=True) #Use for gray scale images
            im = cv2.resize(im,(40,40))
            train.append(im)
            label.append(count)
        count+=1
                        
    train = np.asarray(train)
    label = np.asarray(label)
    X,y = shuffle(train,label,random_state = 10)
    if save == True:
        np.save('X',X)
        np.save('y',y)
    return X,y

    

if __name__ == "__main__" :
    filepath = 'C:/Users/Aniket/Desktop/Aniket/data/qsar_aquatic_toxicity.csv'
    imagedir = 'C:/Users/Aniket/Desktop/Aniket/data/chest_xray/val'
    start_col = 0
    end_col = 7
    target_col = 8
    #X, y = csv_to_numpy(filepath,end_col, target_col)
    #df = csv_to_dataframe(filepath)
    #X,y = imagedir_to_numpy(imagedir)

    print(X,y)
    print('done')
