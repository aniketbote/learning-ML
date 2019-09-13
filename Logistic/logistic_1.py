import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.preprocessing import scale
from sklearn.linear_model import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn import preprocessing

#load data and see
address=("ex2data2.txt")
student=pd.read_csv(address)
student.columns=['score1','score2','res']
#print(student.head())

#seperate input and output data
X=student.ix[:,(0,1)].values
student_data_names=['score1','score2']

y=student.ix[:,2].values
#print("all done")

#check missing values
#print(student.isnull().sum())

#check if output contains other than 0 or 1
#plt.show(sb.countplot(x='res', data=student))

#print(student.info())

X = scale(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

LogReg=DecisionTreeClassifier()
LogReg.fit(X_train,y_train)
dat1=LogReg.predict(X_test)

print(r2_score(dat1,y_test))
cm=confusion_matrix(y_test,dat1)
plt.figure(figsize=(10,5))
plt.show(sb.heatmap(cm,annot=True))
plt.xlabel('predicted')
plt.ylabel('Truth')



