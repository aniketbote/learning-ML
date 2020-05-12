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

def demo(col, data):
    feture_layer = tf.keras.layers.DenseFeatures(col)
    print(feture_layer(data).numpy())

# logdir = "logs"
# file_writer = tf.summary.create_file_writer(logdir)
# file_writer.set_as_default()

def scheduler(epoch):
    if epoch < 10:
        lr =  0.001
    else:
        lr =  0.0001
    tf.summary.scalar(name = 'learning rate', data=lr, step=epoch)
    return lr


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed = 121 )
    ds = ds.batch(batch_size)
    return ds


def create_model():
    model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(units = 12, activation='relu', use_bias = True, kernel_initializer= 'glorot_uniform', bias_initializer = 'zeros'),
    tf.keras.layers.Dense(units = 6, activation='relu', use_bias = True, kernel_initializer= 'glorot_uniform', bias_initializer = 'zeros'),
    tf.keras.layers.Dense(units = 2, activation='softmax')
    ])
    return model



BATCH_SIZE = 32 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train_data, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(val_data, shuffle=False, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(test_df, shuffle=False, batch_size=BATCH_SIZE)

example_batch = next(iter(train_ds))[0]


print('\n'*10)
# for feature_batch, label_batch in train_ds.take(1):
#     print('Every feature:', list(feature_batch.keys()))
#     print('A batch of ages:', feature_batch['age'])
#     print('A batch of targets:', label_batch )






pre_age = Preprocess('age')
age_num = pre_age.num_column()
age_buck = pre_age.buck_column([20, 30, 40, 50])
demo(age_num, example_batch)

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
# print(feature_layer(example_batch).numpy())



model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])




ALL_CALLBACKS = []
csv_logger = tf.keras.callbacks.CSVLogger('training.csv', append = False)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose = 1)
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=1
    )
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    min_delta = 0,
    verbose = 1,
    mode = 'min'
    )
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'my_model/mymodel_{epoch}_{val_loss:.2f}.h5',
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    mode = 'min')
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


preds = model.predict(test_ds)

print(preds)
