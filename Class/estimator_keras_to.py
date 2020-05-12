import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
import seaborn as sns



train_df = pd.read_csv('mytrain2.csv')
test_df = pd.read_csv('mytest2.csv')
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=133, shuffle= True)

class Preprocess:
    def __init__(self, name):
        self.name = name
    def num_column(self):
        nc = tf.feature_column.numeric_column(self.name)
        return nc
    def buck_column(self, buckets):
        nc = self.num_column()
        bc = tf.feature_column.bucketized_column(nc, boundaries = buckets)
        return bc
    def cat_column(self, vocab):
        cc = tf.feature_column.categorical_column_with_vocabulary_list(self.name, vocab)
        cc_i = tf.feature_column.indicator_column(cc)
        return cc_i
    def emb_column(self, vocab, dimension):
        cc = tf.feature_column.categorical_column_with_vocabulary_list(self.name, vocab)
        ec = tf.feature_column.embedding_column(cc, dimension = dimension)
        return ec


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function


def create_model(feature_layer):
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 12, activation='relu', use_bias = True, kernel_initializer= 'glorot_uniform', bias_initializer = 'zeros'),
    tf.keras.layers.Dense(units = 6, activation='relu', use_bias = True, kernel_initializer= 'glorot_uniform', bias_initializer = 'zeros'),
    tf.keras.layers.Dense(units = 2, activation='softmax')
    ])
    return model

pre_age = Preprocess('age')
age_num = pre_age.num_column()
age_buck = pre_age.buck_column([20, 30, 40, 50])
# demo(age_num, example_batch)

pre_thal = Preprocess('thal')
thal_vocab = train_df['thal'].unique()
thal_one_hot = pre_thal.cat_column(thal_vocab)
thal_emb = pre_thal.emb_column(thal_vocab, 3)

feature_columns = []
feature_columns.append(age_num)
feature_columns.append(age_buck)
feature_columns.append(thal_one_hot)
feature_columns.append(thal_emb)



feature_layer = tf.keras.layers.DenseFeatures(feature_columns)





model = create_model(feature_layer)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#Creating estimator from keras model
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model = model,
    model_dir = 'keras_estimator_model',
    custom_objects = feature_layer
    )
# print(train_data.head)
train_label = train_data.pop('target')
val_label = val_data.pop('target')

train_input = make_input_fn(train_data, train_label)
val_input = make_input_fn(val_data, val_label)

keras_estimator.train(input_fn=train_input, steps=10000)
eval_result = keras_estimator.evaluate(input_fn=val_input, steps=10)
print('Eval result: {}'.format(eval_result))
