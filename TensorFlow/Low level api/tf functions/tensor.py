import tensorflow as tf

#constant value Declarations
tensor1 = tf.constant([1, 2, 3, 4, 5, 6], dtype = tf.int64)
tensor2 = tf.constant([6, 2, 3, 4, 10, 6], shape=(4,3), dtype = tf.int64)
tensor3 = tf.constant(-1, shape=[4, 3], dtype = tf.int64)
tensor4 = tf.constant([3, 2, 7, 10, 10, 4], shape=(3,4), dtype = tf.int64)
tensor5 = tf.constant([1.0, 2.0, 3.0], dtype = tf.float32)



a = tf.constant(3.0, dtype = tf.float32)
b = tf.constant(4.0, dtype = tf.float32)
c = tf.constant(5.0, dtype = tf.float32)
#Dynamic value Declarations
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#Uninitialized variables
uni = tf.get_variable('uni',initializer = tf.constant(1.0,dtype = tf.float32))
temp = tf.get_variable('temp',[1,2,3])


#Start session
sess = tf.Session()

#Initializing all uninitialized variables
sess.run(tf.global_variables_initializer())

#If else in tensorflow
'''
if (x < y && x > z) raise OpError("Only one predicate may evaluate to True");
if (x < y) return 17;
else if (x > z) return 23;
else return -1;
'''
def f1(): return tf.constant(17)
def f2(): return tf.constant(23)
def f3(): return tf.constant(-1)
r = tf.case({tf.less(c, b): f1, tf.greater(c, a): f2},
         default=f3, exclusive=True)
print(sess.run(r))

#Displaying the tensors
print(sess.run(uni))
print(sess.run(tensor4))

#operations and outputs
total = a + b
mult = tensor2 *tensor3
print(sess.run({'mult':mult, 'total':total}))

z = x + y
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

sorted_tensor = tf.argsort(tensor2, axis = -1, direction = 'ASCENDING')
print(sess.run(sorted_tensor))

max_index = tf.math.argmax(tensor1)
print(sess.run(max_index))

min_index = tf.math.argmin(tensor1)
print(sess.run(min_index))

assign_value = tf.assign(uni,a)
print(sess.run(assign_value))

add_assign = tf.assign_add(uni,a)
print(sess.run(add_assign))

sub_assign = tf.assign_sub(uni,a)
print(sess.run(sub_assign))

clip_global = tf.clip_by_global_norm([tensor5],a)
print(sess.run(clip_global))

clip_norm = tf.clip_by_norm(tensor5,a)
print(sess.run(clip_norm))

clip_value = tf.clip_by_value(tensor5,1.0,2.0)
print(sess.run(clip_value))

matrix_mult_1 = tf.matmul(tensor2,tensor4)
print(sess.run(matrix_mult_1))

index_add = min_index * max_index
matrix_mult_2 = tf.matmul((tensor2*tensor3),tensor4)
final = index_add * matrix_mult_2
print(sess.run(final))

#Creating Event file for tensorboard
writer = tf.summary.FileWriter('event')
writer.add_graph(tf.get_default_graph())
writer.flush()
