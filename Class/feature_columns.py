import tensorflow as tf
from tensorflow.keras import layers
a = {}
a['X'] = [-1,2,7,3,8,9,23,90,10,-47]

c = {}
c['X'] = ['zero','one','one','zero','zero','zero','one']
def demo_numeric(feature_column):
    global a
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(a).numpy())

def demo_cat(feature_column):
    global c
    feature_layer = layers.DenseFeatures([tf.feature_column.indicator_column(feature_column)])
    print(feature_layer(c).numpy())

def demo_emb(feature_column):
    global c
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(c).numpy())

b = tf.feature_column.numeric_column('X')
demo_numeric(b)

bucketized_price = tf.feature_column.bucketized_column(b, boundaries=[0,10,50])
demo_numeric(bucketized_price)

vocab = set(c['X'])
cat = tf.feature_column.categorical_column_with_vocabulary_list('X', vocab)
demo_cat(cat)

emb = tf.feature_column.embedding_column(cat, dimension=2)
demo_emb(emb)
