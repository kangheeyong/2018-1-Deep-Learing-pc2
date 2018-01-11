import tensorflow as tf
import numpy as np
import os
import sys
import my_lib 
import time 
os.environ["CUDA_VISIBLE_DEVICES"]="3"

 
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


def simple_D(x,isTrain=True,reuse = False) :
    
    with tf.variable_scope('D', reuse=reuse) :
        
        #x = (-1,28,28,1)
        conv1 = tf.layers.conv2d(x,64,[5,5], strides=(1,1),padding = 'valid')
        r1 = tf.nn.elu(tf.layers.batch_normalization(conv1,training=isTrain))#64*24*24

   
        conv2 = tf.layers.conv2d(r1,128,[5,5], strides=(1,1),padding = 'valid')
        r2 = tf.nn.elu(tf.layers.batch_normalization(conv2,training=isTrain))#128*20*20

  
        conv3 = tf.layers.conv2d(r2,256,[5,5], strides=(1,1),padding = 'valid')
        r3 = tf.nn.elu(tf.layers.batch_normalization(conv3,training=isTrain))#256*16*16

 
        conv4 = tf.layers.conv2d(r3,512,[4,4], strides=(2,2),padding = 'same')
        r4 = tf.nn.elu(tf.layers.batch_normalization(conv4,training=isTrain))#512*8*8


        conv5 = tf.layers.conv2d(r4,1024,[4,4], strides=(2,2),padding = 'same')
        r5 = tf.nn.elu(tf.layers.batch_normalization(conv5,training=isTrain))#1024*4*4

       
        conv6 = tf.layers.conv2d(r5,2048,[4,4], strides=(1,1),padding = 'valid')
        r6 = tf.nn.elu(conv6)#2048*1*1

 
        conv7 = tf.layers.conv2d(r6,1,[1,1], strides=(1,1),padding = 'valid')
        r7 = tf.nn.sigmoid(conv7)#1*1*1


        return r7



u = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='u')
ref = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='ref')
t = tf.placeholder(tf.float32, shape = (None, 28,28,1), name='t')
isTrain = tf.placeholder(dtype=tf.bool,name='isTrain') 
 

G_y = simple_G(u,ref,isTrain=isTrain,name='G_y')

    

D_real = simple_D(t,isTrain)
D_fake = simple_D(G_y,isTrain,reuse=True)

D_loss =  tf.reduce_mean(-0.5*(tf.log(D_real + 1e-8) + tf.log(1-D_fake + 1e-8)),name='D_loss')
gan_loss =  tf.reduce_mean(-0.5*(tf.log(D_fake + 1e-8)),name='gan_loss')
 
  
cross_entropy = tf.reduce_mean(0.5*(-t*tf.log(G_y + 1e-8) - (1-t)*tf.log(1-G_y + 1e-8)),name='cross_entropy')
mse = tf.reduce_mean(tf.square(t-G_y)/2,name='mse') 

content_loss = tf.reduce_mean(tf.abs(t-G_y),name='content_loss')

G_loss = tf.add(0.9*content_loss,0.1*gan_loss,name='G_loss')

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('D')]
G_vars = [var for var in T_vars if var.name.startswith('G')]

# When using the batchnormalization layers,
# it is necessary to manually add the update operations
# because the moving averages are not included in the graph
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) :    
    D_optim = tf.train.AdamOptimizer(0.0001).minimize(D_loss, var_list=D_vars, name='D_optim') 
    G_optim = tf.train.AdamOptimizer(0.0001).minimize(G_loss, var_list=G_vars, name='G_optim')






sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

np.random.seed(1)

test_images = mnist.test.images[0:16]    
test_origin = test_images*0.5
test_ref,_ = mnist.test.next_batch(16)
test_input = np.minimum(test_origin + test_ref*0.7 + np.random.uniform(size = (16,784)), 1.0)

np.random.seed(int(time.time()))


my_lib.mnist_4by4_save(np.reshape(test_input,(-1,784)),file_name + '/input_noise.png')
my_lib.mnist_4by4_save(np.reshape(test_origin,(-1,784)),file_name + '/ground_true.png') 
my_lib.mnist_4by4_save(np.reshape(test_ref,(-1,784)),file_name + '/input_ref.png')

hist_G=[]
hist_D=[]

G_error = []
D_error = []
gan_error = []
content_errpr=[]




log_txt = open(file_name +'/log.txt','w')
for i in range(200000) :
    train_images,_ = mnist.train.next_batch(100)
    train_origin = train_images * np.random.uniform(0.2,1.0)
    train_ref,_ = mnist.train.next_batch(100)
    train_input = np.minimum(train_origin + train_ref*np.random.uniform(0.2,1.0) + np.random.uniform(size = (100,784)), 1.0)
    
    _ , D_e = sess.run([D_optim, D_loss],feed_dict={u : np.reshape(train_input,(-1,28,28,1)),
        ref : np.reshape(train_ref,(-1,28,28,1)), t : np.reshape(train_origin,(-1,28,28,1)),
        isTrain : True})
    D_error.append(D_e) 

#    train_images,_ = mnist.train.next_batch(100)
#    train_origin = train_images * np.random.uniform(0.2,1.0)
#    train_ref,_ = mnist.train.next_batch(100)
#    train_input = np.minimum(train_origin + train_ref*np.random.uniform(0.2,1.0) + np.random.uniform(size = (100,784)), 1.0)
    
    _ , G_e, gan_e, content_e= sess.run([G_optim, G_loss, gan_loss, content_loss],
        feed_dict={u : np.reshape(train_input,(-1,28,28,1)), ref : np.reshape(train_ref,(-1,28,28,1)),
        t : np.reshape(train_origin,(-1,28,28,1)),isTrain : True})
    G_error.append(G_e)
    gan_error.append(gan_e)
    content_errpr.append(content_e)


    if i%1000 == 0:
        hist_D.append(np.mean(D_error))
        hist_G.append(np.mean(G_error))
  
        r,val_e = sess.run([G_y,content_loss],feed_dict={u : np.reshape(test_input,(-1,28,28,1)),
            ref : np.reshape(test_ref,(-1,28,28,1)), t : np.reshape(test_origin,(-1,28,28,1)),
            isTrain : False})
        
        print('D_e : %.4f, G_e : %.4f, g_e : %.4f, c_e : %.4f, v_e : %.4f'%(np.mean(D_error),
            np.mean(G_error), np.mean(gan_error), np.mean(content_errpr),val_e))
        log_txt.write('D_e : %.4f, G_e : %.4f, g_e : %.4f, c_e : %.4f, v_e : %.4f\n'%(np.mean(D_error),
            np.mean(G_error), np.mean(gan_error), np.mean(content_errpr),val_e))
        my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/result_{}.png'.format(str(i).zfill(3)))
        G_error = []
        D_error = []
        gan_error = []
        content_errpr=[]



 
my_lib.gan_loss_graph_save(G_loss = hist_G,D_loss=hist_D,path = file_name + '/loss_graph.png')
saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')


end = time.time()-start

print("total time : ",end)
log_txt.write('total time : %f'%(end))

log_txt.close()











