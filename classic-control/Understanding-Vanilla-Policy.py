import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

tf.reset_default_graph()

sess = tf.Session()
state_in = tf.placeholder(shape=[None, 2], dtype=tf.float32)
hidden = slim.fully_connected(state_in, 4, biases_initializer=None, activation_fn=tf.nn.relu)
output = slim.fully_connected(hidden, 2, biases_initializer=None, activation_fn = tf.nn.softmax)

exp = tf.range(0, tf.shape(output)[0]) * tf.shape(output)[1] + [1]

init = tf.global_variables_initializer()
sess.run(init)

# Vanilla Policy Car Pole's agent gradient function feed_dict is vstacked,
# each state is going to be 1-d array(by reshape), then gather final output action tensor's value from indexes.
# Finally, responsible_outputs could use for reduce_mean process

print( sess.run(exp, feed_dict={state_in:[[2,3],[1,2]]}))
print( sess.run(tf.reshape(output, [-1]), feed_dict={state_in:[[2,3],[1,2]]}))

# [1 3]
# [0.891988   0.108012   0.79760593 0.20239407]

sess.close()