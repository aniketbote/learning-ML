from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

def createDataset(fname,start_col,end_col,target_col):
    columns = list(range(start_col,end_col))
    data = np.genfromtxt('../Dataset/original/{}'.format(fname), delimiter=',',skip_header = 1,usecols=columns)
    labels = np.genfromtxt('../Dataset/original/{}'.format(fname), delimiter=',',skip_header = 1,usecols=(target_col),dtype=int)
    return data,labels



dataSet,labels= createDataset('train.csv',2,202,1)

print('dataset created')
#norm = 'Original'
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


train,lol = autonorm(dataSet)
print('dataset normalized')
train_labels = labels
x_train, x_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.2, random_state=42)




model = Sequential()
model.add(Dense(units=1064, activation='tanh'))
model.add(Dense(units=3200, activation='sigmoid'))
model.add(Dense(units=6400, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)

classes = list(model.predict(x_test))
final = []
for i in classes:
  #print(i)
  y_classes = i.argmax(axis=-1)
  #print(y_classes)
  final.append(y_classes)
print('done')

from sklearn.metrics import f1_score
score = f1_score(y_test,final,average = None)
print(score)
