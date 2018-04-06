
import numpy as np
import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b

writer = tf.summary.FileWriter('my_graph')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()
print(sess.run(total))