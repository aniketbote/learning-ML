from sklearn import datasets
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import (RandomUnderSampler, 
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)
import numpy as np
#import pandas as pd
from collections import Counter
#import matplotlib.pyplot as plt
#import scipy
#from scipy.stats import spearmanr
import sklearn
#from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import r2_score
#from sklearn import preprocessing
from sklearn.metrics import f1_score
import time
#from sklearn.decomposition import PCA
#import pickle
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


X, y = datasets.make_classification(
            n_samples     = 16000,  # number of data points
            n_classes     = 2,      # number of classes
            n_clusters_per_class=2, # The number of clusters per class 
            weights       = [0.8,0.2], # The proportions assigned to each class
            n_features    = 10,     # number of total features 
            n_informative = 2,      # number of informative features 
            n_redundant   = 2,      # number of redundant features
            random_state  = 0                       )

print('Original dataset shape {}'.format(Counter(y)))
print(y[:10])
ossampler =  SMOTE(ratio ={0:15000,1:15000})
X, y = ossampler.fit_sample(X, y)
print(y[:10])
print('Oversampled dataset shape {}'.format(Counter(y)))

rat = [{0:10000,1:10000},{0:2500,1:10000},{0:10000,1:2500}]
sampler = RandomUnderSampler(ratio={0:2500,1:10000})
X_rs, y_rs = sampler.fit_sample(X, y)
print('Random undersampling {}'.format(Counter(y_rs)))


##from imblearn.datasets import make_imbalance
##X_rs, y_rs = make_imbalance(X, y, sampling_strategy={1: 1000, 0: 65},
##                      random_state=0)
##print('Random undersampling {}'.format(Counter(y_rs)))
##print(X_rs.shape)
##functions=[RandomUnderSampler,NearMiss,NeighbourhoodCleaningRule,TomekLinks,ClusterCentroids]
##function_names = ['RandomUnderSampler','NearMiss','NeighbourhoodCleaningRule','TomekLinks','ClusterCentroids']
##rat=[1.0,0.8,0.6,0.4,0.2,{1:3000,0:1000},{1:3000,0:1500},{1:1000,0:1000},{1:3000,0:3000}]
##osratio=[1.0,0.7,0.5,0.3,{1:15000,0:15000}]
##for k in osratio:
##        print(k)
##        ossampler =  RandomOverSampler(ratio = k,random_state=0)
##        X_train_norm, y_train_norm = ossampler.fit_sample(X, y)
##        X_train_org, y_train_org = ossampler.fit_sample(X, y)
##        print(Counter(y_train_norm))
##
##
##        for fn,fn_name in zip(functions,function_names):
##            for i in ['_norm','_org']:
##                for j in rat:            
##                    sampler = fn(ratio=j)
##                    print(j)
##                    X_train = eval('X_train{}'.format(i))
##                    y_train = eval('y_train{}'.format(i))
##                    try:
##                        X_rs, y_rs = sampler.fit_sample(X_train, y_train)
##                    except:
##                        continue
##                    #print('--------------------222')
##                    #print('Random undersampling {}'.format(Counter(y_rs)))
##
##                    LogReg=LogisticRegression()
##                    LogReg.fit(X_rs, y_rs)
##
##                    X_test =eval('X_test{}'.format(i))
##                    predictions=list(LogReg.predict(X_test))
##                    y_test = eval('y_test{}'.format(i))
##                    score1 = f1_score(y_test, predictions,average='macro')
##                    score2 = f1_score(y_test,predictions,average=None)
##                    if i == '_norm':
##                        NORM = 'Normalized'
##                    else:
##                        NORM = 'Orginal'
##                    score_dict={}
##                    #score_dict['over_sampler']=str(os_name)
##                    #score_dict['over_ratio']=k
##                    score_dict['under_sampler']=str(fn_name)
##                    score_dict['under_ratio']=j
##                    score_dict['{}'.format(NORM)]=score2
##                    score_dict['macro']=score1
##                    all_scores.append(score_dict)
##                    final_all_scores.append(score_dict)
##                    print(score_dict)
##
##
####
##


##sampler = RandomUnderSampler(ratio=0.8)
##X_rs, y_rs = sampler.fit_sample(X, y)
##print('Random undersampling {}'.format(Counter(y_rs)))
##
##
##
### ClusterCentroids
##sampler = ClusterCentroids(ratio={1: 100, 0: 100})
##X_rs, y_rs = sampler.fit_sample(X, y)
##print('Cluster centriods undersampling {}'.format(Counter(y_rs)))
##
##
### TomekLinks
##sampler = TomekLinks(ratio={1: 100, 0: 100},random_state=0)
##X_rs, y_rs = sampler.fit_sample(X, y)
##print('TomekLinks undersampling {}'.format(Counter(y_rs)))
##
##
##
### NeighbourhoodCleaningRule
##sampler = NeighbourhoodCleaningRule(ratio={1: 100, 0: 100},random_state=0)
##X_rs, y_rs = sampler.fit_sample(X, y)
##print('NearestNeighbours Clearning Rule undersampling {}'.format(Counter(y_rs)))
##
##
### NearMiss
##sampler = NearMiss(ratio={1: 100, 0: 100},random_state=0)
##X_rs, y_rs = sampler.fit_sample(X, y)
##print('NearMiss{}'.format(Counter(y_rs)))
##

### SMOTE Over Sampling
##sampler = SMOTE(ratio=1.0,random_state=0)
##X_rs, y_rs = sampler.fit_sample(X, y)
##print('Over Sampling{}'.format(Counter(y_rs)))
##
##
##
##sampler =  RandomOverSampler(ratio=1.0,random_state=0)
##X_rs, y_rs = sampler.fit_sample(X, y)
##print('Random Over Sampling{}'.format(Counter(y_rs)))
##

