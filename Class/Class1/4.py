import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
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

# on_(train|test|predict)_begin
# on_(train|test|predict)_end
# on_(train|test|predict)_batch_begin
# on_(train|test|predict)_batch_end
# on_epoch_begin
# on_epoch_end

class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, name):
        self.name = name

    def on_train_begin(self, batch, logs=None):
        # print('training begin')
        # print(batch)
        # print(logs)
        pass
        # print('\n Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    def on_train_batch_begin(self, batch, logs=None):
        # print('train batch begin')
        # print(batch)
        # print(logs)
        pass
        # print('\n Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
    def on_epoch_begin(self, epoch, logs = None):
        # print('train batch begin')
        # print(logs)
        # print(epoch)
        pass
    def on_epoch_end(self, epoch, logs=None):
        # print(self.validation_data)
        print('train epoch end')
        print(logs)
        print(epoch)
        print(self.name)
        pass
        # print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

    def on_train_batch_end(self, batch, logs=None):
        pass
        # print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

    def on_test_batch_begin(self, batch, logs=None):
        pass
        # print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    def on_test_batch_end(self, batch, logs=None):
        pass
        # print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))



def demo(col, data):
    feture_layer = tf.keras.layers.DenseFeatures(col)
    print(feture_layer(data).numpy())

logdir = "logs"
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

def scheduler(epoch):
    if epoch <= 100:
        learning_rate =  0.01
    else:
        learning_rate = 0.001
    tf.summary.scalar(name = 'learning rate', data=learning_rate, step=epoch)
    return learning_rate


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
    tf.keras.layers.Dense(units = 12, activation='relu', use_bias = True, kernel_initializer= 'identity', bias_initializer = 'zeros', name = 'd1'),
    tf.keras.layers.Dense(units = 6, activation='relu', use_bias = True, kernel_initializer= 'glorot_normal', bias_initializer = 'zeros', name = 'd2'),
    tf.keras.layers.Dense(units = 3, activation='softmax', name = 'out')
    ])
    return model



BATCH_SIZE = 10 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train_data, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(val_data, shuffle=False, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(test_df, shuffle=False, batch_size=BATCH_SIZE)

example_batch = next(iter(train_ds))[0]


print('\n'*10)
# for feature_batch, label_batch in train_ds.take(1):
#     print('Every feature:', list(feature_batch.keys()))
#     print('A batch of ages:', feature_batch['age'])
#     print('A batch of targets:', label_batch )


# age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
NUMERIC_COLUMNS = ['sex','trestbps','chol','fbs','thalach','exang','oldpeak']
CATEGORICAL_COLUMNS = ['cp','restecg','slope','ca','thal']
feature_columns = []

pre_age = Preprocess('age')
age_buck = pre_age.buck_column([20, 30, 40, 50])
feature_columns.append(age_buck)

for col in NUMERIC_COLUMNS:
    temp_obj = Preprocess(col)
    num_col = temp_obj.num_column()
    feature_columns.append(num_col)

for col in CATEGORICAL_COLUMNS:
    temp_obj = Preprocess(col)
    vocab = train_df[col].unique()
    one_hot = temp_obj.cat_column(vocab)
    feature_columns.append(one_hot)





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
    verbose = 0)
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
    patience=10,
    min_delta = 0.00001,
    verbose = 0,
    mode = 'min'
    )
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'my_model/mymodel_{epoch}_{val_loss:.2f}.h5',
    monitor = 'val_loss',
    verbose = 0,
    save_best_only = True,
    mode = 'min')
lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedule = scheduler, verbose = 0)



ALL_CALLBACKS.append(csv_logger)
ALL_CALLBACKS.append(tensorboard)
ALL_CALLBACKS.append(lr_schedule)
ALL_CALLBACKS.append(reduce_lr)
# ALL_CALLBACKS.append(model_checkpoint)
ALL_CALLBACKS.append(MyCustomCallback('Aniket'))


history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=10000,
          verbose = 1,
          callbacks = ALL_CALLBACKS)

# print(model.summary())

print(history.history)


loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
print("Loss", loss)


# model.save('mymodel')

# new_model = tf.keras.models.load_model('mymodel')
# history = new_model.fit(train_ds,
#           validation_data=val_ds,
#           epochs=5,
#           verbose = 1,
#           callbacks = ALL_CALLBACKS)


preds = model.predict(test_ds)
# print(preds)
print(tf.math.argmax(preds, 1).numpy())
