import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
import seaborn as sns
import os

tf.executing_eagerly()

train_df = pd.read_csv('mytrain2.csv')
test_df = pd.read_csv('mytest2.csv')
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=133, shuffle= True)
test_label = test_df['target']

bool_pos = train_data['target'] == 1
train_data_pos = train_data[bool_pos].reset_index(drop = True)
train_data_neg = train_data[~bool_pos].reset_index(drop = True)

count_neg = len(train_data_neg)



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


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed = 121 )
    ds = ds.batch(batch_size)
    return ds


def make_dataset(dataframe, shuffle=True, batch_size=32):
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=100000, seed = 121 ).repeat()
    return ds


def create_model():
    feature_columns, feature_layer_inputs = create_input()
    dense_features = tf.keras.layers.DenseFeatures(feature_columns)(feature_layer_inputs)
    hidden1 = tf.keras.layers.Dense(units = 12, activation='relu')(dense_features)
    hidden2 = tf.keras.layers.Dense(units = 6, activation='relu')(hidden1)
    output1 = tf.keras.layers.Dense(units = 2, activation='softmax')(hidden2)
    model = tf.keras.models.Model(
            inputs = [v for v in feature_layer_inputs.values()],
            outputs = output1)

    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



BATCH_SIZE =  32# A small batch sized is used for demonstration purposes
pos_ds = make_dataset(train_data_pos)
neg_ds = make_dataset(train_data_neg)
train_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed = 45)
train_ds = train_ds.batch(BATCH_SIZE)
steps_per_epoch = np.ceil(2.0*count_neg/BATCH_SIZE)



val_ds = df_to_dataset(val_data, shuffle=False, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(test_df, shuffle=False, batch_size=BATCH_SIZE)



def create_input():
    NUMERIC_COLUMNS = ['sex','trestbps','chol','fbs','thalach','exang','oldpeak']
    CATEGORICAL_COLUMNS = ['cp','restecg','slope','ca']
    feature_columns = []
    feature_layer_inputs = {}


    pre_age = Preprocess('age')
    age_buck = pre_age.buck_column([20, 30, 40, 50])
    feature_columns.append(age_buck)
    feature_layer_inputs['age'] = tf.keras.Input(shape=(1,), name='age')

    for col in NUMERIC_COLUMNS:
        temp_obj = Preprocess(col)
        num_col = temp_obj.num_column()
        feature_columns.append(num_col)
        feature_layer_inputs[col] = tf.keras.Input(shape=(1,), name=col)

    for col in CATEGORICAL_COLUMNS:
        temp_obj = Preprocess(col)
        vocab = train_df[col].unique()
        one_hot = temp_obj.cat_column(vocab)
        feature_columns.append(one_hot)
        feature_layer_inputs[col] = tf.keras.Input(shape=(1,), name=col, dtype=tf.dtypes.int64)
    return feature_columns, feature_layer_inputs


model1 = create_model()



ALL_CALLBACKS = []

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=1
    )



ALL_CALLBACKS.append(tensorboard)

history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=1000000,
          verbose = 1,
          steps_per_epoch=steps_per_epoch,
          callbacks = ALL_CALLBACKS)


loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
print("Loss", loss)





preds = model.predict(test_ds)
predictions = tf.math.argmax(preds, 1).numpy()


cm = tf.math.confusion_matrix(
    test_label, predictions,
    num_classes=2
)
print(cm.numpy())
sns.heatmap(cm, annot = True)
# plt.show()

result = tf.keras.metrics.Precision()
result.update_state(test_label, predictions)
print('precison score is: ', result.result().numpy()) # 0.57142866
