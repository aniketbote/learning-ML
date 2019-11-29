import tensorflow as tf
import time
from keras.utils import np_utils


mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train, X_test = X_train / 255.0, X_test / 255.0

n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)



model1 = tf.keras.models.Sequential([
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='ones',
                 bias_initializer='zeros'),
   tf.keras.layers.Dense(10, activation='softmax',kernel_initializer='ones',
                 bias_initializer='zeros')
 ])


model1.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

x=time.time()
model1.fit(X_train, Y_train, epochs=5,batch_size=3000,verbose = 1)
y=time.time()
tft = y-x
tfs = model1.evaluate(X_test, Y_test,batch_size = 3000)




from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils


model = Sequential()
model.add(Dense(128, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(1024, activation='relu', kernel_initializer='ones',
                bias_initializer='zeros'))
model.add(Dense(10,activation='softmax', kernel_initializer='ones',
                bias_initializer='zeros'))



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


x=time.time()
model.fit(X_train, Y_train, epochs=5,batch_size=3000)
y=time.time()
kt = y-x
ks = model.evaluate(X_test, Y_test,batch_size=3000)


print(model1.metrics_names)
print(tfs)
print(tft)

print(model.metrics_names)
print(ks)
print(kt)

writer = tf.summary.FileWriter('event')
writer.add_graph(tf.get_default_graph())
writer.flush()
