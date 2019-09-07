import numpy as np
import pandas as pd
from collections import Counter
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
from operator import itemgetter
from pprint import pprint
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import (RandomUnderSampler, 
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)
import warnings
warnings.filterwarnings("ignore")



#load data and see
def createDataset(fname,start_col,end_col,target_col):
    columns = list(range(start_col,end_col))
    data = np.genfromtxt('Dataset/{}'.format(fname),max_rows=100, delimiter=',',skip_header = 1,usecols=columns)
    labels = np.genfromtxt('Dataset/{}'.format(fname),max_rows=100, delimiter=',',skip_header = 1,usecols=(target_col),dtype=int)
    return data,labels

#normalization
def autonorm(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    range_vals = minvals - maxvals
    norm_data = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    norm_data = dataset - np.tile(minvals,(m,1))
    norm_data = norm_data/np.tile(range_vals,(m,1))
    return norm_data



dataSet,labels= createDataset('train.csv',2,202,1)

norm_data = autonorm(dataSet)
org_data = dataSet
train_labels = labels
X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(norm_data, train_labels, test_size=0.2, random_state=42)
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(org_data, train_labels, test_size=0.2, random_state=42)
print(Counter(y_train_norm))
df = pd.DataFrame(X_train_org)
df['output']=y_train_org
df.to_csv('output.csv',index=None)
ossampler =  RandomOverSampler(ratio =1.0)
X, y = ossampler.fit_sample(X_train_org, y_train_org)
df = pd.DataFrame(X)
df['output']=y
df.to_csv('output1.csv',index=None)

print(Counter(y))

rat = [{0:10000,1:10000},{0:2500,1:10000},{0:10000,1:2500}]
functions=[RandomUnderSampler,NearMiss,NeighbourhoodCleaningRule,TomekLinks,ClusterCentroids]
function_names = ['RandomUnderSampler','NearMiss','NeighbourhoodCleaningRule','TomekLinks','ClusterCentroids']
os_fn = [SMOTE,RandomOverSampler]
os_names=['SMOTE','RandomOverSampler']
