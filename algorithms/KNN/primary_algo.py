import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score
import operator


def autonorm(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    range_vals = minvals - maxvals
    norm_data = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    norm_data = dataset - np.tile(minvals,(m,1))
    norm_data = norm_data/np.tile(range_vals,(m,1))
    return norm_data


def classify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1))- dataSet #tile repeats the array n times
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



data = pd.read_csv('primary_dataset.csv')
df = pd.DataFrame(data)
#print(list(df))
unwanted_columns = ['score_id', 'child_id', 'scorer_id', 'video_file', 'updatedAt', 'question_set', 'age(months)', 'Male', 'createdAt', 'updatedAt.1', 'Unnamed: 41']
for names in unwanted_columns:
    del df['{}'.format(str(names))]

y_true = list(df['diag'])
possible_values = list(set(y_true))
for i in range(0,len(y_true)):
    if y_true[i]==possible_values[0]:
        y_true[i]=0
    else:
        y_true[i]=1
#print(y_true)

del df['diag']
df['asd']=y_true
#print(df.shape)
for names in list(df):
    df = df[pd.notnull(df['{}'.format(names)])]
#print(df.head())

examples = df.values
train = examples[:,0:29]
train_labels = examples[:,30] 

X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.1, random_state=4)


#prediction = []
#start=time.time()
for k in range(1,21):
    prediction = []
    start=time.time()
    for i in range(0,int(X_test.shape[0])):
        #print(i)
        #temp=[]
        #temp.append(test_list[i])
        #prediction_value = classify(X_test[i],X_train,y_train,1)
        #temp.append(prediction_value)
        #prediction.append(temp)
        prediction.append(classify(X_test[i],X_train,y_train,k))#k= 4 0.899 k=5 9.015


    #print(len(prediction))
    #print(prediction)
    score = accuracy_score(y_test, prediction, normalize=True)
    end=time.time()
    print(k,score)
#print(end-start)


##
##prediction = []
##start=time.time()
##for i in range(0,int(X_test.shape[0])):
##    #print(i)
##    #temp=[]
##    #temp.append(test_list[i])
##    #prediction_value = classify(X_test[i],X_train,y_train,1)
##    #temp.append(prediction_value)
##    #prediction.append(temp)
##    prediction.append(classify(X_test[i],X_train,y_train,7))#k= 4 0.899 k=5 9.015
##
##
###print(len(prediction))
###print(prediction)
##score = accuracy_score(y_test, prediction, normalize=True)
##end=time.time()
##print(score)




#print(train[0:5])

#print(train.shape)
#print(train_labels)
