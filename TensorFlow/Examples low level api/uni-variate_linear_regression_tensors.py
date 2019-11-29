#Time difference
'''
#For random generated data using python numpy operations
    epochs = 10000
    n = 1000000
    lr = 0.0001    
    time = 2487.6255564170306 s
    w = 0.1660430736673414 b = 0.37184915765539467

#For random generated data using tensorflow tensors operation
    epochs = 10000
    n = 1000000
    lr = 0.0001    
    time = 61.28795552253723 s
    w = 0.1660430736673414  b =  0.37184915765539467


#For random generated data using tensorflow tensors operation(GPU 1050Ti)
    epochs = 10000
    n = 1000000
    lr = 0.0001    
    time = 15.528876304626465 s
    w = 0.1660430736673414  b =  0.37184915765539467


'''
#Python code for gradient descent
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)
X = np.asarray(df['X'], dtype = np.float64)
Y = np.asarray(df['Y'], dtype = np.float64)
n = float(len(X))

m = 0
c = 0
L = 0.0001  # The learning Rate
epochs = 10000  # The number of iterations to perform gradient descent
n = float(len(X)) # Number of elements in X


# Performing Gradient Descent
start_time = time.time()
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
 
end_time = time.time()
print(end_time-start_time)

print (m, c)    
pred = m*X + c

plt.scatter(X,Y)
plt.plot([min(X), max(X)], [min(pred), max(pred)], color='red')
plt.show()

'''
#Gradient descent using Tensors

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split    
from sklearn.metrics import r2_score
from scipy.interpolate import make_interp_spline, BSpline

#Parameters
epochs = 1000
lr = 0.000001
m = 0
c = 0
losses = []

#Loading data
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)
X = np.asarray(df['X'], dtype = np.float64)
Y = np.asarray(df['Y'], dtype = np.float64)

###Spliting into train_test
##X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=43)

#no of samples
n = float(len(X))

###loading random numbers
##np.random.seed(1)
##X = np.asarray(np.random.rand(1000000))
##np.random.seed(10)
##Y = np.asarray(np.random.rand(1000000))
##n = float(len(X))

#Tf constants
x = tf.constant(X, dtype = tf.float64)
y = tf.constant(Y, dtype = tf.float64)
learning_rate = tf.constant(lr, dtype=tf.float64)

#Tf variables
m = tf.compat.v1.get_variable('m', initializer = tf.constant(m,dtype = tf.float64))
c = tf.compat.v1.get_variable('c', initializer = tf.constant(c,dtype = tf.float64))
d_m = tf.compat.v1.get_variable('d_m', initializer = tf.constant(0.0,dtype = tf.float64))
d_c = tf.compat.v1.get_variable('d_c', initializer = tf.constant(0.0,dtype = tf.float64))
y_pred = tf.compat.v1.get_variable('y_pred', initializer = tf.constant(0.0,dtype = tf.float64))

#Graph of gradient descent
y_pred = m*x + c
loss = tf.reduce_sum(tf.compat.v1.losses.mean_squared_error(y, y_pred))
d_m = (-2/n) * tf.reduce_sum(x*(y-y_pred)) 
d_c = (-2/n) * tf.reduce_sum(y-y_pred)  
upm = tf.compat.v1.assign(m, m - learning_rate * d_m)
upc = tf.compat.v1.assign(c, c - learning_rate * d_c)

#starting session
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


#Training for epochs
start_time = time.time()
for i in range(epochs):
    losses.append(sess.run(loss))
    sess.run([upm,upc])
end_time = time.time()
print(end_time-start_time)

#Retrieving m and c
w,b = sess.run([m,c])
print(w,b)

#Predictions
pred = w*X + b

#Accuracy
print(r2_score(Y,pred))

#Plotting learned line
plt.scatter(X,Y)
plt.plot([min(X), max(X)], [min(pred), max(pred)], color='red')
plt.show()

#plotting loss curve
ep = np.asarray(range(1,epochs+1))
xnew = np.linspace(ep.min(),ep.max(),300) #300 represents number of points to make between T.min and T.max
spl = make_interp_spline(ep, losses, k=3) #BSpline object
power_smooth = spl(xnew)
plt.plot(xnew,power_smooth)
plt.show()

#Creating event file for tensorboard
writer = tf.compat.v1.summary.FileWriter('event')
writer.add_graph(tf.compat.v1.get_default_graph())
writer.flush()
