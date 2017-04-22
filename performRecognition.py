
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np
import tensorflow as tf
import os 

#Tensorflow model  variables

n_input = 784 #inputsize
n_classes = 10 #outputsize
dropout = 0.75 
learning_rate = 0.001
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) 



def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # Maxpool wrapper
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

pred = conv_net(x, weights, biases, keep_prob)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing variables
init = tf.global_variables_initializer()



# Read the input image 
im = cv2.imread("photo_1.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
_,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
saver=tf.train.Saver()

# Restoring the tensorflow model
with tf.Session() as sess:
         sess.run(init)
         filename=r'model.ckpt'
         f=os.path.realpath(__file__)
         path=os.path.join(os.path.dirname(os.path.abspath(f)),filename)
         saver.restore(sess,path)

# For each rectangular region,predict the digit using convnet model.
         nbr=[]
         k=[]
         for rect in rects:
             # Draw the rectangles
             cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
             # Make the rectangular region around the digit
             leng = int(rect[3] * 1.6)
             pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
             pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
             roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
             
             # Resize the image
             roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
             
             
             nbr.append(roi)
             _x=np.reshape(roi,(-1,784))
             _x=_x.astype(np.float32)
             
             k.append(sess.run(tf.argmax(pred,1), feed_dict={x:_x,keep_prob: 1.}))
         #plt.ion()
         
         #plt.show()
         #m=1
         for i in k:
             #m=0
             fig=plt.figure()
             a=fig.add_subplot(1,2,1)
             #m=m+1
             img=plt.imread("photo_1.jpg")
             imgplot=plt.imshow(img)
             #m=m+1
             #fig=plt.figure()
             a=fig.add_subplot(1,2,2)
             img=nbr[k.index(i)]
             imgplot=plt.imshow(img)
             a.set_title(i[0])
             plt.colorbar(ticks=[0.1,0.3,0.5,0.7],orientation='horizontal')
         plt.show() 
    