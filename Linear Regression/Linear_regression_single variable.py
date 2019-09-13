import numpy as np
import pandas as pd
import seaborn as sb
from pylab import rcParams
from sklearn.model_selection import train_test_split    
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from collections import Counter
from sklearn.metrics import r2_score

address='dat.txt'
data=pd.read_csv(address)
data.columns=['profits','population']
#plt.show(sb.pairplot(data))

X=data.ix[:,0].values
y=data.ix[:,1].values
z=plt.scatter(X,y)
#X,y=scale(X),y
#print(X,y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train=X_train.reshape(-1,1)
reg=LinearRegression()
reg.fit(X_train,y_train)
X_test=X_test.reshape(-1,1)
y_pred=reg.predict(X_test)
print(r2_score(y_test,y_pred))
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    z=plt.plot(x_vals, y_vals, '--')

abline(reg.coef_,reg.intercept_)
plt.show(z)
