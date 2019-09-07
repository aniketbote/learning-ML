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
from collections import Counter
from imblearn.under_sampling import (RandomUnderSampler, 
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)


data = pd.read_csv('C:/Users/Aniket/Desktop/Aniket/ML/Dataset/original/train.csv')
df = pd.DataFrame(data)
array = np.array(df)

X_main = array[:,2:]
y_main = array[:,1]
y_main = y_main.astype('int')

print(Counter(y_main))
# RandomUnderSampler
sampler = RandomUnderSampler(ratio=1.0)
X, y = sampler.fit_sample(X_main, y_main)
print(X.shape)
print(y.shape)
a = np.append(X,y[:,None],axis=1)

final_df =  pd.DataFrame(a)
final_df.to_csv('undersampled_random.csv')
print(Counter(y))

# ClusterCentroids
sampler = ClusterCentroids(ratio=1.0)
X, y = sampler.fit_sample(X_main, y_main)
print(X.shape)
print(y.shape)
b = np.append(X,y[:,None],axis=1)

final_df =  pd.DataFrame(b)
final_df.to_csv('undersampled_cluster.csv')
print(Counter(y))


# NearMiss
sampler = NearMiss(ratio=1.0)
X, y = sampler.fit_sample(X_main, y_main)
print(X.shape)
print(y.shape)
c = np.append(X,y[:,None],axis=1)

final_df =  pd.DataFrame(c)
final_df.to_csv('undersampled_nearmiss.csv')
print(Counter(y))

##fdf = pd.DataFrame()
##for i in range(X.shape[1]):
##    temp = X[:,i]
##    mean = np.mean(temp)
##    temp1 = temp.reshape(1,-1)
##    bn = Binarizer(threshold=mean)
##    pd_watched = bn.transform(temp1)[0]
##    fdf['{}'.format(i)] = pd_watched
##    print(i)
##X = np.array(fdf)
##
##
##
##
##
##
##
##X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state= 43)
##
##y_train = y_train.astype('int')
##y_test = y_test.astype('int')
##
##
##knn = KNeighborsClassifier(n_neighbors=5)
##knn.fit(X_train, y_train)
##filename = 'knn_1.sav'
##pickle.dump(knn, open(filename, 'wb'))
##score = knn.score(X_test, y_test)
##print(score)
