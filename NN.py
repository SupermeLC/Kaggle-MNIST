import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
#===============================
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#===============================
learning_rate = 0.3
epoche = 5000
in_units = 784
hl_units1 = 300
hl_units2 = 300
#===============================

w1 = tf.Variable(tf.truncated_normal([in_units, hl_units1], stddev = 0.1))#标准差0.1
b1 = tf.Variable(tf.zeros([hl_units1]))
w2 = tf.Variable(tf.truncated_normal([hl_units1, hl_units2], stddev = 0.1))
b2 = tf.Variable(tf.zeros([hl_units2]))
w3 = tf.Variable(tf.zeros([hl_units2, 10]))
b3 = tf.Variable(tf.zeros([10]))

#dropout
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)


hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, w2) + b2)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

y = tf.nn.softmax(tf.matmul(hidden2_drop, w3) + b3)
#y = tf.nn.softmax(tf.matmul(hidden1, w2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy) #学习率为0.5

#===================================

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#===================================
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(epoche):
    ##
    batch_xs, batch_ys = mnist.train.next_batch(100)

    ##
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.8})
    if i % 100 == 0:
        print(i,"  ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


#accuracy = 97.







