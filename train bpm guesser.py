import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

import Generate_Data

save_path = "D:\\Downloads\\BPM Identifier\\Saved models\\model.ckpt"
save_rate = 2000

x_data, y_data = Generate_Data.loadTrainingData()


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.40

batch_size = 5


x = tf.placeholder(tf.float32, shape=[None, 40000])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                        strides=[1, 1, 2, 1], padding='SAME')

W_conv1 = weight_variable([1, 256, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 1, 40000, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 8, 1],
                        strides=[1, 1, 8, 1], padding='SAME')

W_conv2 = weight_variable([1, 256, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([2500 * 64, 512])
b_fc1 = bias_variable([512])

h_pool2_flat = tf.reshape(h_pool2, [-1, 2500*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512, 1])
b_fc2 = bias_variable([1])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

loss = tf.reduce_mean(
    tf.losses.absolute_difference(labels=y_, predictions=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

def plot(loss_list, predictions, batchX, batchY):
    plt.subplot(1, 3, 1)
    plt.cla()
    plt.plot(loss_list)


    plt.subplot(1, 3, 3)
    plt.cla()
    plt.plot(batchY, color="red")
    plt.plot(predictions, color="green")

    plt.draw()
    plt.pause(0.0001)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if input("Do you want to restore the model?(y/n): ") == "y":
        saver.restore(sess, save_path)
        print("Model restored.")

    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []
  
    for i in range(40000):
        if i % 4000 == 0:
            x_data = np.random.permutation(x_data)
            y_data = np.random.permutation(y_data)
        start_idx = (i % 4000) * batch_size
        end_idx = start_idx + batch_size

        batchX = x_data[start_idx:end_idx,:]
        batchY = y_data[start_idx:end_idx,:]

	
        if i % 100 == 0:
            train_loss, _predictions = sess.run([loss, y_conv], feed_dict={
		  x: batchX, y_: batchY, keep_prob: 1.0})
            loss_list.append(train_loss)
            print('step %d, training loss %g' % (i, train_loss))
            plot(loss_list, _predictions, batchX, batchY)

        if i%save_rate == 0:
                print("Saving model")
                saver.save(sess, save_path) 
            
        train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})

    print('test loss %g' % loss.eval(feed_dict={
            x: x_data[0:10,:], y_: y_data[0:10,:], keep_prob: 1.0}))
    
    _save_path = saver.save(sess, save_path)
    print("Model saved in file: %s" % _save_path)

plt.ioff()
plt.show()
