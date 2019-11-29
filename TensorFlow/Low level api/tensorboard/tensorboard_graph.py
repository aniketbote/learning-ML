import tensorflow as tf
a = tf.constant(3, dtype = tf.int32)
b = tf.constant(4, dtype = tf.int32)
total = a + b
sess = tf.Session()
print(sess.run(total))
writer = tf.summary.FileWriter('event')
writer.add_graph(tf.get_default_graph())
writer.flush()
