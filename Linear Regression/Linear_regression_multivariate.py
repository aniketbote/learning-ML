import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

data=pd.read_csv("data2.txt")
data.columns=['size','no of rooms','price']
plt.show(sb.heatmap(data))
X,y=scale(data.ix[:,0:2]),data.ix[:,2]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
reg=LinearRegression()
reg.fit(X_train,y_train)
dat1=reg.predict(X_test)

print(r2_score(y_test,dat1))


