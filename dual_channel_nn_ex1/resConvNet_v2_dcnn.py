import tensorflow as tf
import numpy as np
import os
import sys
import my_lib 
import time 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

 
start = time.time()

 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('.')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)

def simple_G(x,ref,isTrain = True, name = 'y') : #input = (minibatch * w * h * ch)
    
    # out size = (in size + 2*padding - kenel)/strides + 1    

    with tf.variable_scope('G') :
    

        cat1 = tf.concat([x,ref],3)
        conv1 = tf.layers.conv2d(cat1,128,[3,3], strides=(1,1),padding = 'same') 
        r1 = tf.nn.elu(tf.layers.batch_normalization(conv1,training=isTrain))
    
        conv2 = tf.layers.conv2d(r1,128,[3,3], strides=(1,1),padding = 'same')
        r2 = tf.nn.elu(tf.layers.batch_normalization(conv2,training=isTrain)) + r1
        
        conv3 = tf.layers.conv2d(r2,128,[3,3], strides=(1,1),padding = 'same')
        r3 = tf.nn.elu(tf.layers.batch_normalization(conv3,training=isTrain)) + r2

        conv4 = tf.layers.conv2d(r3,128,[3,3], strides=(1,1),padding = 'same')
        r4 = tf.nn.elu(tf.layers.batch_normalization(conv4,training=isTrain)) + r3

        conv5 = tf.layers.conv2d(r4,128,[3,3], strides=(1,1),padding = 'same')
        r5 = tf.nn.elu(tf.layers.batch_normalization(conv5,training=isTrain)) + r4

        conv6 = tf.layers.conv2d(r5,2,[3,3], strides=(1,1),padding = 'same')
        r6 = tf.nn.elu(tf.layers.batch_normalization(conv6,training=isTrain)) + cat1
        
        conv7 = tf.layers.conv2d(r6,1,[3,3], strides=(1,1),padding = 'same')
    r7 = tf.nn.sigmoid(conv7,name=name)
   
    return r7





u = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='u')
ref = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='ref')
t = tf.placeholder(tf.float32, shape = (None, 28,28,1), name='t')
isTrain = tf.placeholder(dtype=tf.bool,name='isTrain') 
 

G_y = simple_G(u,ref,isTrain=isTrain,name='G_y')

  
cross_entropy = tf.reduce_mean(0.5*(-t*tf.log(G_y + 1e-8) - (1-t)*tf.log(1-G_y + 1e-8)),name='cross_entropy')
mse = tf.reduce_mean(tf.square(t-G_y)/2,name='mse') 

content_loss = tf.reduce_mean(tf.abs(t-G_y),name='content_loss')


T_vars = tf.trainable_variables()
G_vars = [var for var in T_vars if var.name.startswith('G')]

# When using the batchnormalization layers,
# it is necessary to manually add the update operations
# because the moving averages are not included in the graph
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) :    
    content_optim = tf.train.AdamOptimizer(0.0001).minimize(content_loss, var_list=G_vars, name='content_optim')




sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

np.random.seed(int(time.time()))

test_images = mnist.test.images[0:16]    
test_origin = test_images*0.5
test_ref,_ = mnist.test.next_batch(16)
test_input = np.minimum(test_origin + test_ref*0.7 + np.random.uniform(size = (16,784)), 1.0)
test_ref = test_ref*0.85

my_lib.mnist_4by4_save(np.reshape(test_input,(-1,784)),file_name + '/input_noise.png')
my_lib.mnist_4by4_save(np.reshape(test_origin,(-1,784)),file_name + '/ground_true.png') 
my_lib.mnist_4by4_save(np.reshape(test_ref,(-1,784)),file_name + '/input_ref.png')
 
log_txt = open(file_name +'/log.txt','w')

content_error = []
mse_error = []
cross_entropy_error = []


for i in range(1000000) :
    
    train_images,_ = mnist.train.next_batch(100)
    train_origin = train_images * np.random.uniform(0.2,1.0)
    train_ref,_ = mnist.train.next_batch(100)
    train_input = np.minimum(train_origin + train_ref*np.random.uniform(0.2,1.0) + np.random.uniform(size = (100,784)), 1.0)
    train_ref = train_ref*np.random.uniform(0.2,1.0)

    _ , content_e, mse_e, cross_entropy_e= sess.run([content_optim, content_loss,mse,cross_entropy],
        feed_dict={u : np.reshape(train_input,(-1,28,28,1)), ref : np.reshape(train_ref,(-1,28,28,1)),
        t : np.reshape(train_origin,(-1,28,28,1)),isTrain : True})
    content_error.append(content_e)
    mse_error.append(mse_e)
    cross_entropy_error.append(cross_entropy_e)

    if i%1000 == 0:
        
        
        r,val_e = sess.run([G_y,content_loss],feed_dict={u : np.reshape(test_input,(-1,28,28,1)),
            ref : np.reshape(test_ref,(-1,28,28,1)), t : np.reshape(test_origin,(-1,28,28,1)),
            isTrain : False})
  
 
        print('%8d,c_e : %.4f, mse : %.6f, cross_entropy : %.4f, val_e : %4f'%(i, np.mean(content_error)
            ,np.mean(mse_error), np.mean(cross_entropy_error),val_e))
        log_txt.write('%8d,c_e : %.4f, mse : %.4f, cross_entropy : %.4f, val_e : %4f\n'%(i, np.mean(content_error)
            ,np.mean(mse_error), np.mean(cross_entropy_error),val_e))
        my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/result_{}.png'.format(str(i).zfill(3)))
 
    content_error = []
    mse_error = []
    cross_entropy_error = []

       
saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')


end = time.time()-start

print("total time : ",end)
log_txt.write('total time : %f'%(end))

log_txt.close()







