import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
#import seaborn as sb
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.metrics import f1_score
import time
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
import pickle
from imblearn.under_sampling import (RandomUnderSampler, 
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)

import warnings
warnings.filterwarnings("ignore")


##x = pd.read_csv('Dataset/test.csv')
##df = pd.DataFrame(x)
##test_list = df['ID_code'].tolist()
##
##final_df = pd.DataFrame()
##final_df['ID_code']=test_list


#load data and see
def createDataset(fname,start_col,end_col,target_col):
    columns = list(range(start_col,end_col))
    data = np.genfromtxt('Dataset/{}'.format(fname),max_rows=20000, delimiter=',',skip_header = 1,usecols=columns)
    labels = np.genfromtxt('Dataset/{}'.format(fname),max_rows=20000, delimiter=',',skip_header = 1,usecols=(target_col),dtype=int)
    return data,labels


dataSet,labels= createDataset('train.csv',2,202,1)

norm = 'Original'
def autonorm(dataset):
    norm = 'Normalized'
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    range_vals = minvals - maxvals
    norm_data = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    norm_data = dataset - np.tile(minvals,(m,1))
    norm_data = norm_data/np.tile(range_vals,(m,1))
    return norm_data,norm

##
##test,waste = createDataset('test.csv',1,201,0)
##test = autonorm(test)



##pca = PCA(n_components=25).fit(dataSet)
##dataSet = pca.transform(dataSet)

#print(dataSet.shape)

#train,norm = autonorm(dataSet)
train = dataSet
train_labels = labels
X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.2, random_state=42)

#print(Counter(y_train))

#without sampling
#Original = [0.95195801 0.37258348]
#Normalized = [0.95392881 0.38198198]
rat = {1: 1500, 0: 6000}

#Oversampling
sampler = SMOTE(ratio={1:15000,0:15000},random_state=0)
X_rs, y_rs = sampler.fit_sample(X_train, y_train)
print('Over Sampling{}'.format(Counter(y_rs)))

###Random sampler
###Ratio =ratio={1: 1500, 0: 4500}     Norm = [0.93591362 0.47410817] [0.93475816 0.47747748]
###Ratio =ratio={1: 1500, 0: 1500}     Norm = [0.84405397 0.38868389] [0.84256789 0.38428484]
###Ratio = {1: 1500, 0: 2000}   Normalized = [0.87602304 0.41654779]
###Ratio = {1: 1500, 0: 4000}   Normalized = [0.92959343 0.47183847]
###Ratio = {1: 1500, 0: 4000}   Original = [0.92689851 0.46073298]
###Ratio = {1: 1500, 0: 6000}   Original = [0.9409154  0.46075949]
###Ratio = {1: 1500, 0: 6000}   Normalized = [0.94262408 0.4589309 ]
###Ratio = {1: 1500, 0: 8000}   Normalized = [0.94864791 0.44542773]
###Ratio = {1: 1500, 0: 8000}   Original = [0.94752706 0.45363766]
###Ratio = {1: 1500, 0: 10000}   Original = [0.95089528 0.42356688]
##sampler = RandomUnderSampler(ratio=rat)
##X_rs, y_rs = sampler.fit_sample(X_train, y_train)
##print('Random undersampling {}'.format(Counter(y_rs)))



### TomekLinks
###Ratio = {1: 1500, 0: 1500}   Original = [0.95304534 0.35897436]
###Ratio = {1: 1500, 0: 3000}   Original = [0.95304534 0.35897436]
###Ratio = {1: 1500, 0: 6000}   Original = [0.95304534 0.35897436]
###Ratio = {1: 1500, 0: 6000}   Normalized = [0.95392881 0.38198198]
###Ratio = {1: 1500, 0: 1500}   Normalized = [0.95369749 0.3715847 ]
##sampler = TomekLinks(ratio=rat,random_state=0)
##X_rs, y_rs = sampler.fit_sample(X_train, y_train)
##print('TomekLinks undersampling {}'.format(Counter(y_rs)))


### NeighbourhoodCleaningRule
###Ratio = {1: 1500, 0: 1500}   Normalized = [0.95392881 0.38198198]
###Ratio = {1: 1500, 0: 4500}   Normalized = [0.95392881 0.38198198]
###Ratio = {1: 1500, 0: 4500}   Original = [0.95195801 0.37258348]
##sampler = NeighbourhoodCleaningRule(ratio=rat,random_state=0)
##X_rs, y_rs = sampler.fit_sample(X_train, y_train)
##print('NearestNeighbours Clearning Rule undersampling {}'.format(Counter(y_rs)))


### NearMiss
###Ratio = {1: 1500, 0: 1500}   Normalized = [0.85982748 0.39655172]
###Ratio = {1: 1500, 0: 1500}   Original = [0.85811332 0.39658569]
###Ratio = {1: 1500, 0: 4500}   Original = [0.93434131 0.45274212]
###Ratio = {1: 1500, 0: 4500}   Normalized = [0.9376656  0.46079614]
###Ratio = {1: 1500, 0: 6000}   Normalized = [0.94639912 0.46132597]
###Ratio = {1: 1500, 0: 6000}   Original = [0.94320851 0.46133683]
###Ratio = {1: 1500, 0: 9000}   Original = [0.9502928  0.44444444]
###Ratio = {1: 1500, 0: 9000}   Normalized = [0.95150115 0.44131455]
###Ratio = {1: 1500, 0: 11000}   Normalized = [0.95216216 0.41      ]
##sampler = NearMiss(ratio=rat,random_state=0)
##X_rs, y_rs = sampler.fit_sample(X_train, y_train)
##print('NearMiss{}'.format(Counter(y_rs)))



##x =time.time()
### ClusterCentroids
###Ratio = {1: 1500, 0: 1500}   Normalized = [0.83667984 0.38328358]
###Ratio = {1: 1500, 0: 1500}   Original = [0.824      0.37142857]
###Ratio = {1: 1500, 0: 4500}   Original = [0.91925287 0.45961538]
###Ratio = {1: 1500, 0: 4500}   Normalized = [0.92948536 0.48654244]
###Ratio = {1: 1500, 0: 6000}   Original = [0.93583637 0.46867749]
###Ratio = {1: 1500, 0: 6000}   Normalized = [0.94019703 0.45649433]
##
##sampler = ClusterCentroids(ratio=rat)
##X_rs, y_rs = sampler.fit_sample(X_train, y_train)
##print('Cluster centriods undersampling {}'.format(Counter(y_rs)))
##y=time.time()
##print(y-x)

#print('training')
LogReg=LogisticRegression()
start = time.time()
LogReg.fit(X_rs, y_rs)
end = time.time()
#print(end-start)


##filename = 'log_normalize.sav'
##pickle.dump(LogReg, open(filename, 'wb'))


#print('predicting')
x =time.time()
dat1=list(LogReg.predict(X_test))
y=time.time()
#print(y-x)
##final_df['target'] = dat1
##print('calculating')



#final_df.to_csv('log_normalize.csv')

score = f1_score(y_test, dat1,average=None)
print(score)

print(Counter(dat1))
print('Ratio = {}   {} = {}'.format(rat,norm,score))
##print(r2_score(dat1,y_test))
##cm=confusion_matrix(y_test,dat1)
##plt.figure(figsize=(10,5))
##plt.show(sb.heatmap(cm,annot=True))
##plt.xlabel('predicted')
##plt.ylabel('Truth')
##
##
##
