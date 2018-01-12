import tensorflow as tf
import numpy as np
import os
import sys
import time
import my_lib 
import time 

os.environ["CUDA_VISIBLE_DEVICES"]="1"

#session 구조에 대해서 공부하기!!!

start = time.time()

  
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('.')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)
    
    
sess_1 = tf.InteractiveSession()


new_saver = tf.train.import_meta_graph('ex_basic_5_v1/para.cktp.meta')
new_saver.restore(sess_1, tf.train.latest_checkpoint('ex_basic_5_v1/'))
    
 
one_hot = np.eye(10)


z = sess_1.graph.get_tensor_by_name("z:0")
#u = sess.graph.get_tensor_by_name("u:0")
z_c = sess_1.graph.get_tensor_by_name("z_c:0")
#z_fill = sess.graph.get_tensor_by_name("z_fill:0")
isTrain = sess_1.graph.get_tensor_by_name("isTrain:0")

    
G_z = sess_1.graph.get_tensor_by_name("G_z:0")

#D_loss = sess.graph.get_tensor_by_name("D_loss:0")
#G_loss = sess.graph.get_tensor_by_name("G_loss:0")

#D_optim = sess.graph.get_operation_by_name("D_optim")
#G_optim = sess.graph.get_operation_by_name("G_optim")


tf.reset_default_graph()

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


cnn_u = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='cnn_u')
cnn_t = tf.placeholder(tf.float32, shape = (None, 10), name='cnn_t')
cnn_y = simple_cnn(cnn_u)

cnn_loss = tf.reduce_mean(-0.5*(cnn_t*tf.log(cnn_y + 1e-8)+ (1-cnn_t)*tf.log(1-cnn_y+1e-8)),name='cnn_loss')
cnn_optim = tf.train.AdamOptimizer(0.0001).minimize(cnn_loss,name='cnn_optim')
    
cnn_correct_prediction = tf.equal(tf.argmax(cnn_y,1),tf.argmax(cnn_t,1))
cnn_accuracy = tf.reduce_mean(tf.cast(cnn_correct_prediction, tf.float32),name = 'cnn_accuracy')







sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())


test_input = np.reshape(mnist.test.images,(-1, 28, 28, 1))
test_label = mnist.test.labels


np.random.seed(int(time.time()))




log_txt = open(file_name +'/log.txt','w')


for i in range(500000) :

  
    cnn_img, cnn_label = mnist.train.next_batch(50)
    cnn_img = np.reshape(cnn_img,(-1,28,28,1))
    cnn_label = np.reshape(cnn_label,(-1,10))

    cnn_z = np.random.normal(0,1,size=(50,1,1,100))
    cnn_temp = np.random.randint(0,9,(50,1))
    cnn_z_c = one_hot[cnn_temp].reshape([-1,1,1,10])
     
    gen_d = sess_1.run([G_z],feed_dict={z : cnn_z,z_c : cnn_z_c, isTrain : False})        
     
    cnn_input = np.concatenate([cnn_img,np.reshape(gen_d,(-1,28,28,1))],0)
    cnn_target = np.concatenate([cnn_label,np.reshape(cnn_z_c,(-1,10))],0)
 

    _ , e, acc = sess.run([ cnn_optim, cnn_loss, cnn_accuracy],
            feed_dict={cnn_u : cnn_input, cnn_t : cnn_target})
   
    if i%1000 == 0:

        test_acc = sess.run([cnn_accuracy],feed_dict={cnn_u : test_input, cnn_t : test_label})
     
        print('cnn_t_e : %.2f, cnn_t_acc : %.2f, test acc : %.6f'%(e,acc,test_acc[0]))
        log_txt.write('cnn_t_e : %.2f, cnn_t_acc : %.2f, test acc : %.6f\n'%(e,acc,test_acc[0]))


        np.random.seed(int(time.time()))

saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')

log_txt.close()

end = time.time()-start

print("total time : ",end)
















