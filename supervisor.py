import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

epsilon = tf.constant(1e-10)

W1 = tf.Variable(tf.random_normal([784, 999]))
b1 = tf.Variable(tf.random_normal([999]))
h1 = tf.matmul(xs, W1) + b1

W2 = tf.Variable(tf.random_normal([999, 10]))
b2 = tf.Variable(tf.random_normal([10]))
y = tf.matmul(h1, W2) + b2

loss = -1*tf.reduce_mean(tf.multiply(tf.log(tf.nn.softmax(y) + epsilon), ys))

global_step = tf.Variable(0, name='global_step')
train_op = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss, global_step=global_step)

'''
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(100):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		print(sess.run([train_op, loss], feed_dict={xs: batch_xs, ys:batch_ys}))
'''

global_step = tf.Variable(0, name='global_step')
train_op = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss, global_step=global_step)

sv = tf.train.Supervisor(
	logdir='./log',
	saver=tf.train.Saver(),
	global_step=global_step,
	save_summaries_secs=1,
	save_model_secs=1,
	checkpoint_basename='test_model.ckpt'
)

with sv.managed_session() as sess:
	for step in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		print(sess.run([train_op, loss], feed_dict={xs: batch_xs, ys:batch_ys}))
