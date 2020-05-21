import numpy as np
from numpy import tile
from pprint import pprint
import operator
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def autonorm(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    range_vals = minvals - maxvals
    norm_data = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    norm_data = dataset - tile(minvals,(m,1))
    norm_data = norm_data/tile(range_vals,(m,1))
    return norm_data

def createDataset(fname,start_col,end_col,target_col):
    columns = list(range(start_col,end_col))
    data = np.genfromtxt('Dataset/{}'.format(fname),max_rows = 100, delimiter=',',skip_header = 1,usecols=columns)
    labels = np.genfromtxt('Dataset/{}'.format(fname),max_rows =100, delimiter=',',skip_header = 1,usecols=(target_col),dtype=int)
    return data,labels


dataSet,labels= createDataset('train.csv',2,202,1)


train = autonorm(dataSet)
train_labels = labels
cdict={0:'blue',1:'red'}


pca = PCA(n_components=3).fit(train)
pca_2d = pca.transform(train)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for g in np.unique(train_labels):
    ix = np.where(train_labels == g)
    ax.scatter(pca_2d[ix,0],pca_2d[ix,1],pca_2d[ix,2],c = cdict[g],label=g,marker='.')
plt.show()

