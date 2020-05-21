import numpy as np
from numpy import tile
from pprint import pprint
import operator
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
##x = pd.read_csv('Dataset/test.csv')
##df = pd.DataFrame(x)
##test_list = df['ID_code'].tolist()
##print(len(test_list))

def createDataset(fname,start_col,end_col,target_col):
    columns = list(range(start_col,end_col))
    data = np.genfromtxt('Dataset/{}'.format(fname),max_rows=10000, delimiter=',',skip_header = 1,usecols=columns)
    labels = np.genfromtxt('Dataset/{}'.format(fname),max_rows=10000, delimiter=',',skip_header = 1,usecols=(target_col),dtype=int)
    return data,labels


dataSet,labels= createDataset('train.csv',2,202,1)


def autonorm(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    range_vals = minvals - maxvals
    norm_data = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    norm_data = dataset - tile(minvals,(m,1))
    norm_data = norm_data/tile(range_vals,(m,1))
    return norm_data


train = autonorm(dataSet)
#train = dataSet
train_labels = labels
X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.2, random_state=42)


print(len(X_train), len(X_test), len(y_train), len(y_test))







#test,waste = createDataset('test.csv',1,201,0)

#print(test.shape[0])



def classify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1))- dataSet #tile repeats the array n times
    #print(diffMat)
    sqDiffMat = diffMat**2
    #print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    #pprint(sqDistances)
    distances = sqDistances**0.5
    #print(distances)
    sortedDist = distances.argsort()
    #print(sortedDist)
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDist[i]]
        #print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        #print(classCount[voteIlabel])
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)


    return sortedClassCount[0][0]
        



prediction = []
for i in range(0,int(X_test.shape[0])):
    print(i)
    #temp=[]
    #temp.append(test_list[i])
    #prediction_value = classify(X_test[i],X_train,y_train,1)
    #temp.append(prediction_value)
    #prediction.append(temp)
    prediction.append(classify(X_test[i],X_train,y_train,1))#k= 4 0.899 k=5 9.015


print(len(prediction))
score = accuracy_score(y_test, prediction, normalize=True)
print(score)
##final = pd.DataFrame(prediction,columns=['ID_code','target'])
##final.to_csv('out.csv')



