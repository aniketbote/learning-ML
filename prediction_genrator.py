import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import Binarizer
import tensorflow as tf
from collections import Counter
x = pd.read_csv('Dataset/original/test.csv')
df = pd.DataFrame(x)
test_list = df['ID_code'].tolist()

final_df = pd.DataFrame()
final_df['ID_code']=test_list

def createDataset(fname,start_col,end_col):
    columns = list(range(start_col,end_col))
    data = np.genfromtxt('Dataset/original/{}'.format(fname), delimiter=',',skip_header = 1,usecols=columns)
    return data

X = createDataset('test.csv',1,201)
print(' loading done')
#preprocessing to be used
################################################################################

filename = 'scaler.sav'
loaded_model = pickle.load(open(filename, 'rb'))
X = loaded_model.transform( X )





print(' prepro done')
##################################################################################

##filename = 'logreq_1.sav'
##loaded_model = pickle.load(open(filename, 'rb'))
##result = list(loaded_model.predict(X))

model = tf.keras.models.load_model('1_3.h5')
classes = list(model.predict(X))
final = []
for i in classes:
  #print(i)
  y_classes = i.argmax(axis=-1)
  #print(y_classes)
  final.append(y_classes)

final_df['target'] = final
print(Counter(final))
final_df.to_csv('submission.csv')
print('done')
