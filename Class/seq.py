import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import seaborn as sns

train_df = pd.read_csv('mytrain2.csv')
test_df = pd.read_csv('mytest2.csv')
train, val = train_test_split(train_df, test_size=0.2, random_state=133, shuffle= True)

# print(train.head)
# plt.hist(train['target'])
# plt.show()


def demo(feature_column):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    # print(feature_layer)
    print(feature_layer(example_batch).numpy())


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed = 122 )
    ds = ds.batch(batch_size)
    return ds

def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.0001


BATCH_SIZE = 5 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(val, shuffle=False, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(test_df, shuffle=False, batch_size=BATCH_SIZE)

print('\n'*10)
# for feature_batch, label_batch in train_ds.take(1):
#     print('Every feature:', list(feature_batch.keys()))
#     print('A batch of ages:', feature_batch['age'])
#     print('A batch of targets:', label_batch )





example_batch = next(iter(train_ds))[0]
# print(example_batch)

age = tf.feature_column.numeric_column("age")
# demo(age)

age_buckets = tf.feature_column.bucketized_column(age, boundaries=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
# demo(age_buckets)

thal_vocab = train_df['thal'].unique()
# print(thal_vocab)

thal = tf.feature_column.categorical_column_with_vocabulary_list(
      'thal', thal_vocab)

thal_one_hot = tf.feature_column.indicator_column(thal)
# demo(thal_one_hot)

thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
# demo(thal_embedding)


feature_columns = []
USE_COLUMNS = ['trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca','sex']


# numeric cols
for header in USE_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(header))


# bucketized cols
feature_columns.append(age_buckets)


# indicator cols
feature_columns.append(thal_one_hot)


# embedding cols
feature_columns.append(thal_embedding)


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
# print(feature_layer(example_batch).numpy())






model = tf.keras.Sequential([
  feature_layer,
  tf.keras.layers.Dense(units = 12, activation='relu', use_bias = True, kernel_initializer= 'glorot_uniform', bias_initializer = 'zeros'),
  tf.keras.layers.Dense(units = 6, activation='relu', use_bias = True, kernel_initializer= 'glorot_uniform', bias_initializer = 'zeros'),
  tf.keras.layers.Dense(units = 2, activation='softmax')
])





model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])





ALL_CALLBACKS = []
csv_logger = tf.keras.callbacks.CSVLogger('training.csv', append = False)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001, verbose = 1)
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1,
    )
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = 'mymodel_{epoch}_{val_accuracy}.h5', monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')
lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedule = scheduler, verbose = 1)

ALL_CALLBACKS.append(csv_logger)
ALL_CALLBACKS.append(tensorboard)
ALL_CALLBACKS.append(lr_schedule)
ALL_CALLBACKS.append(reduce_lr)
ALL_CALLBACKS.append(model_checkpoint)



model.fit(train_ds,
          validation_data=val_ds,
          epochs=10000,
          verbose = 1,
          callbacks = ALL_CALLBACKS)




loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
print("Loss", loss)
