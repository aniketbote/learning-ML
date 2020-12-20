#!/usr/bin/env python
# coding: utf-8

# In[18]:


import tensorflow as tf
import pandas as pd 
import numpy as np


# In[19]:


x = np.random.randint(100, size =(63319, 7330), dtype = np.int16)
y = np.random.randint(2, size =(63319,), dtype = np.int16)


# In[ ]:


df = pd.DataFrame()
for i in range(x.shape[1]):
    df['col_{}'.format(i)] = x[:,i]


# In[ ]:





# In[14]:


feature_columns = []
for i in range(x.shape[1]):
    feature_columns.append(tf.feature_column.numeric_column('col_{}'.format(i),dtype=tf.float32))


# In[15]:


def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(10000)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(64000)
        return dataset
    return input_fn
train_input_fn = make_input_fn(df, y)
del x


# In[16]:


linear_est = tf.estimator.LinearClassifier(feature_columns, n_classes = 2)


# In[17]:


linear_est.train(train_input_fn, max_steps=1)


# In[ ]:




