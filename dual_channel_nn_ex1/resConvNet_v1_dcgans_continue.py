import tensorflow as tf
import numpy as np
import os
import sys
import my_lib 
import time 
os.environ["CUDA_VISIBLE_DEVICES"]="2"

 
start = time.time()

 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('_continue.py')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)

sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph(file_name + '/para.cktp.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(file_name + '/'))



u = tf.graph.get_tensor_by_name("u:0")
ref = tf.graph.get_tensor_by_name("ref:0")
t = tf.graph.get_tensor_by_name("t:0")
isTrain = tf.graph.get_tensor_by_name("isTrain:0")

G_y = tf.graph.get_tensor_by_name("G_y:0")

gan_loss = tf.graph.get_tensor_by_name("gan_loss:0")
cross_entropy = tf.graph.get_tensor_by_name("cross_entropy:0")
mse = tf.graph.get_tensor_by_name("mse:0")
content_loss = tf.graph.get_tensor_by_name("content_loss:0")


D_loss = sess.graph.get_tensor_by_name("D_loss:0")
G_loss = sess.graph.get_tensor_by_name("G_loss:0")

D_optim = sess.graph.get_operation_by_name("D_optim")
G_optim = sess.graph.get_operation_by_name("G_optim")

np.random.seed(int(time.time()))

test_images,_ = mnist.test.next_batch(16)    
test_origin = test_images*0.5
test_ref,_ = mnist.test.next_batch(16)
test_input = np.minimum(test_origin + test_ref*0.7 + np.random.uniform(size = (16,784)), 1.0)

my_lib.mnist_4by4_save(np.reshape(test_input,(-1,784)),file_name + '/input_noise.png')
my_lib.mnist_4by4_save(np.reshape(test_origin,(-1,784)),file_name + '/ground_true.png') 
my_lib.mnist_4by4_save(np.reshape(test_ref,(-1,784)),file_name + '/input_ref.png')

hist_G=[]
hist_D=[]
log_txt = open(file_name +'/log.txt','w')
for i in range(1000000) :
    G_error = []
    D_error = []
    gan_error = []
    content_errpr=[]

    train_images,_ = mnist.train.next_batch(100)
    train_origin = train_images * np.random.uniform(0.2,1.0)
    train_ref,_ = mnist.train.next_batch(100)
    train_input = np.minimum(train_origin + train_ref*np.random.uniform(0.2,1.0) + np.random.uniform(size = (100,784)), 1.0)
    
    _ , D_e = sess.run([D_optim, D_loss],feed_dict={u : np.reshape(train_input,(-1,28,28,1)),
        ref : np.reshape(train_ref,(-1,28,28,1)), t : np.reshape(train_origin,(-1,28,28,1)),
        isTrain : True})
    D_error.append(D_e) 

    train_images,_ = mnist.train.next_batch(100)
    train_origin = train_images * np.random.uniform(0.2,1.0)
    train_ref,_ = mnist.train.next_batch(100)
    train_input = np.minimum(train_origin + train_ref*np.random.uniform(0.2,1.0) + np.random.uniform(size = (100,784)), 1.0)
    
    _ , G_e, gan_e, content_e= sess.run([G_optim, G_loss, gan_loss, content_loss],
        feed_dict={u : np.reshape(train_input,(-1,28,28,1)), ref : np.reshape(train_ref,(-1,28,28,1)),
        t : np.reshape(train_origin,(-1,28,28,1)),isTrain : True})
    G_error.append(G_e)
    gan_error.append(gan_e)
    content_errpr.append(content_e)


    if i%1000 == 0:
        hist_D.append(np.mean(D_error))
        hist_G.append(np.mean(G_error))
        
        print('D_e : %.4f, G_e : %.4f, gan_e : %.4f, content_e : %.4f'%(np.mean(D_error),
            np.mean(G_error), np.mean(gan_error), np.mean(content_errpr)))
        log_txt.write('D_e : %.4f, G_e : %.4f, gan_e : %.4f, content_e : %.4f\n'%(np.mean(D_error),
            np.mean(G_error), np.mean(gan_error), np.mean(content_errpr)))
 
        r = sess.run([G_y],feed_dict={u : np.reshape(test_input,(-1,28,28,1)),
            ref : np.reshape(test_ref,(-1,28,28,1)), t : np.reshape(test_origin,(-1,28,28,1)),
            isTrain : False})
        my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/result_{}.png'.format(str(i).zfill(3)))


my_lib.gan_loss_graph_save(G_loss = hist_G,D_loss=hist_D,path = file_name + '/loss_graph.png')
saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')


end = time.time()-start

print("total time : ",end)
log_txt.write('total time : %f'%(end))

log_txt.close()











