import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v2.feature_column
import seaborn as sns

import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv('mytrain.csv')
dfeval = pd.read_csv('myeval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')


# print(dftrain.head())
#
# print(dict(dftrain))
#
# print(dftrain.describe())

sns.relplot(x = 'age', y = 'fare', hue = y_train, data = dftrain)
plt.show()



CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))





def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)




ds = make_input_fn(dftrain, y_train, batch_size=10)()

for feature_batch, label_batch in ds.take(1):
  print('Some feature keys:', list(feature_batch.keys()))
  print()
  print('A batch of class:', feature_batch['class'].numpy())
  print()
  print('A batch of Labels:', label_batch.numpy())




age_column = feature_columns[7]
print(tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy())



gender_column = feature_columns[0]
print(tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy())




linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, model_dir = 'models', n_classes = 2, optimizer = tf.keras.optimizers.Adam, warm_start_from = 'models/model.ckpt-400' )
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)


print('\n'*10)
print(result)


# age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)
#
# derived_feature_columns = [age_x_gender]
# linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
# linear_est.train(train_input_fn)
# result = linear_est.evaluate(eval_input_fn)
#
#
#
#
# print('\n'*10)
# print(result)
