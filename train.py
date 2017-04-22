
from __future__ import print_function

import tensorflow as tf
import os
from dataset import mnist
from tensorflow.examples.tutorials.mnist import input_data
data=mnist()
#mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 784 #inputsize
n_classes = 10 #outputsize
dropout = 0.5 

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) 

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    #maxpool wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')



def conv_net(x, weights, biases, dropout):
   
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # conv layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling 
    conv1 = maxpool2d(conv1, k=2)

    # conv layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max pooling
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    #prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#prediction
pred = conv_net(x, weights, biases, keep_prob)

#cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#optimizing with Adamoptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#checking accuracy of prediction
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#initializing tensorflow graph variables
init = tf.global_variables_initializer()
saver=tf.train.Saver() #saver for  saving the model

with tf.Session() as sess:
    sess.run(init)
    step = 1

    #Training
    #for each step calculating loss and accuracy
    while step * batch_size < training_iters:
        #batch=mnist.train.next_batch(batch_size)
        batch_x, batch_y = data.next_batch(batch_size)
        # Running the optimizer 
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        
        
        if step % display_step == 0:
            # for each 1000 steps loss,accuracy are printed 
            
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y,keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    filename=r'model.ckpt'
    f=os.path.realpath(__file__)
    path=os.path.join(os.path.dirname(os.path.abspath(f)),filename)
    
    #Saving model for further use
    save_path=saver.save(sess,path)
    batch_x, batch_y = data.next_batch(256,True)
    #batch=mnist.test.next_batch(256)
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x,y: batch_y,keep_prob: 1.}))