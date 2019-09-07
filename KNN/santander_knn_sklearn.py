from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spstats
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('Dataset/train.csv')
df = pd.DataFrame(data)
array = np.array(df)

X = array[:,2:]
y = array[:,1]
y = y.astype('int')

sampler = SMOTE(ratio=1.0,random_state=0)
X, y = sampler.fit_sample(X, y)

fdf = pd.DataFrame()
for i in range(X.shape[1]):
    temp = X[:,i]
    mean = np.mean(temp)
    temp1 = temp.reshape(1,-1)
    bn = Binarizer(threshold=mean)
    pd_watched = bn.transform(temp1)[0]
    fdf['{}'.format(i)] = pd_watched
    print(i)
X = np.array(fdf)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state= 43)

y_train = y_train.astype('int')
y_test = y_test.astype('int')


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
filename = 'knn_1.sav'
pickle.dump(knn, open(filename, 'wb'))
knn.score(X_test, y_test)
