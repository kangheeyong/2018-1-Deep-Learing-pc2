import tensorflow as tf
import numpy as np
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="2"


   
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('.')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)


def simple_cnn(x) :
    
    w_init = tf.truncated_normal_initializer(mean=0.0, stddev = 0.02)
    b_init = tf.constant_initializer(0.0)

    # out size = (in size + 2*padding - kenel)/strides + 1    

    with tf.variable_scope('cnn') :
        
        
        conv1 = tf.layers.conv2d(x,32,[5,5], strides=(1,1),padding = 'valid')
        r1 = tf.nn.elu(conv1)#32*24*24
        
        conv2 = tf.layers.conv2d(r1,64,[5,5], strides=(1,1),padding = 'valid')
        r2 = tf.nn.elu(conv2)#64*20*20
        
        conv3 = tf.layers.conv2d(r2,128,[5,5], strides=(1,1),padding = 'valid')
        r3 = tf.nn.elu(conv3)#128*16*16
        
        conv4 = tf.layers.conv2d(r3,256,[4,4], strides=(2,2),padding = 'same')
        r4 = tf.nn.elu(conv4)#256*8*8

        conv5 = tf.layers.conv2d(r4,512,[4,4], strides=(2,2),padding = 'same')
        r5 = tf.nn.elu(conv5)#512*4*4
        
        conv6 = tf.layers.conv2d(r5,1024,[4,4], strides=(1,1),padding = 'valid')
        r6 = tf.nn.elu(conv6)#1024*1*1

        fc = tf.reshape(r6,[-1,1024*1*1])       

        # 1st layer
        w1 = tf.get_variable('w1',[fc.get_shape()[1],1024],initializer = w_init)
        b1 = tf.get_variable('b1',[1024],initializer = b_init)
        fc1 = tf.nn.elu(tf.matmul(fc,w1)+b1)

         # 1st layer
        w2 = tf.get_variable('w2',[fc1.get_shape()[1],10],initializer = w_init)
        b2 = tf.get_variable('b2',[10],initializer = b_init)
        fc2 = tf.nn.sigmoid(tf.matmul(fc1,w2)+b2)

        return fc2


u = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='u')
t = tf.placeholder(tf.float32, shape = (None, 10), name='t')
y = simple_cnn(u)

loss = tf.reduce_mean(-0.5*(t*tf.log(y + 1e-8)+(1-t)*tf.log(1-y+1e-8)),name='loss')
optim = tf.train.AdamOptimizer(0.00001).minimize(loss,name='optim')
    
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name = 'accuracy')





sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

test_input = np.reshape(mnist.test.images,(-1, 28, 28, 1))
test_label = mnist.test.labels


log_txt = open(file_name +'/log.txt','w')


for i in range(500000) :
    train_input, train_label = mnist.train.next_batch(100)
    a = np.reshape(train_input,(-1,28,28,1))
                
    _ , e, acc = sess.run([ optim, loss, accuracy],feed_dict={u : a, t : train_label})
    if i % 1000 ==0 :
        test_acc = sess.run([accuracy],feed_dict={u : test_input, t : test_label})
        print('e : %.6f, acc : %.6f, test acc : %.6f'%(e,acc, test_acc[0]))
        log_txt.write('e : %.6f, acc : %.6f, test acc : %.6f\n'%(e,acc, test_acc[0]))



log_txt.close()
saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')








          
