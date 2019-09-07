import numpy as np
import pandas as pd
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

#variables

#load data and see
def createDataset(fname,start_col,end_col,target_col):
    columns = list(range(start_col,end_col))
    data = np.genfromtxt('Dataset/{}'.format(fname),max_rows=200, delimiter=',',skip_header = 1,usecols=columns)
    labels = np.genfromtxt('Dataset/{}'.format(fname),max_rows=200, delimiter=',',skip_header = 1,usecols=(target_col),dtype=int)
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
X_train_1, X_test_norm, y_train_1, y_test_norm = train_test_split(norm_data, train_labels, test_size=0.2, random_state=42)
X_train_2, X_test_org, y_train_2, y_test_org = train_test_split(org_data, train_labels, test_size=0.2, random_state=42)
#rat = {1: 3, 0: 3}
#count = Counter(y_train_norm)
#print(count[1])
rat = [{0:10000,1:10000},{0:2500,1:10000},{0:10000,1:2500}]
functions=[RandomUnderSampler,NearMiss,NeighbourhoodCleaningRule,TomekLinks,ClusterCentroids]
function_names = ['RandomUnderSampler','NearMiss','NeighbourhoodCleaningRule','TomekLinks','ClusterCentroids']
os_fn = [SMOTE,RandomOverSampler]
os_names=['SMOTE','RandomOverSampler']
osratio=[{0:15000,1:15000}]


final_all_scores = []
for os,os_name in zip(os_fn,os_names):
    for k in osratio:
        ossampler = os(ratio = k,random_state=0)
        X_train_norm, y_train_norm = ossampler.fit_sample(X_train_1, y_train_1)
        #print(Counter(y_train_norm))
        X_train_org, y_train_org = ossampler.fit_sample(X_train_2, y_train_2)
        for fn,fn_name in zip(functions,function_names):
            all_scores=[]
            print(str(fn_name))
            for i in ['_norm','_org']:
                for j in rat:            
                    sampler = fn(ratio=j,random_state=0)
                    X_train = eval('X_train{}'.format(i))
                    y_train = eval('y_train{}'.format(i))
                    #try:
                    X_rs, y_rs = sampler.fit_sample(X_train, y_train)
                    print(y_rs.shape)
                    #print('Random undersampling {}'.format(Counter(y_rs)))
                    if i == '_norm':
                        NORM = 'Normalized'
                    else:
                        NORM = 'Orginal'

                    LogReg=LogisticRegression()
                    LogReg.fit(X_rs, y_rs)
                    
                    X_test =eval('X_test{}'.format(i))
                    predictions=list(LogReg.predict(X_test))
                    y_test = eval('y_test{}'.format(i))
                    score1 = f1_score(y_test, predictions,average='macro')
                    score2 = f1_score(y_test,predictions,average=None)


                    df = pd.DataFrame(X_rs)
                    #write_data = np.append(X_rs,y_rs,axis=0)
                    #print(write_data.shape)
                    #df = pd.DataFrame(write_data)
                    df.to_csv('out1.csv')
                    break

                    print('norm={}  rat={}  over={}  {}  {}'.format(NORM,j,k,os_name,fn_name))
                    score_dict={}
                    score_dict['over_sampler']=str(os_name)
                    score_dict['over_ratio']=k
                    score_dict['under_sampler']=str(fn_name)
                    score_dict['under_ratio']=j
                    score_dict['{}'.format(NORM)]=score2
                    score_dict['macro']=score1
                    all_scores.append(score_dict)
                    final_all_scores.append(score_dict)
                    #print(score_dict)
                    #except Exception as e:
                    print('normerrrorerrererer={}  rat={}  over={}  {}  {}'.format(NORM,j,k,os_name,fn_name))
                    print('\n\n')
                    print(e)
                break
                    #continue
            break
            printlist = sorted(all_scores, key=itemgetter('macro'), reverse=True)
            #pprint(printlist[:6])
            
    
print('\n\n---------------------------------------------------')
printlist = sorted(final_all_scores, key=itemgetter('macro'), reverse=True)
#pprint(printlist[:20])







