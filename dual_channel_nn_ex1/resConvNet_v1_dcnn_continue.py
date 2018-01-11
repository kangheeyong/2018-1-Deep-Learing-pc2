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



file_name = sys.argv[0].split('_continue.py')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)

sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph(file_name + '/para.cktp.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(file_name + '/'))


u = sess.graph.get_tensor_by_name("u:0")
ref = sess.graph.get_tensor_by_name("ref:0")
t = sess.graph.get_tensor_by_name("t:0")
isTrain = sess.graph.get_tensor_by_name("isTrain:0")

G_y = sess.graph.get_tensor_by_name("G_y:0")

cross_entropy = sess.graph.get_tensor_by_name("cross_entropy:0")
mse = sess.graph.get_tensor_by_name("mse:0")
content_loss = sess.graph.get_tensor_by_name("content_loss:0")

content_optim = sess.graph.get_operation_by_name("content_optim")




np.random.seed(1)

test_images = mnist.test.images[0:16]   
test_origin = test_images*0.5
test_ref,_ = mnist.test.next_batch(16)
test_input = np.minimum(test_origin + test_ref*0.7 + np.random.uniform(size = (16,784)), 1.0)

np.random.seed(int(time.time()))


my_lib.mnist_4by4_save(np.reshape(test_input,(-1,784)),file_name + '/1_input_noise.png')
my_lib.mnist_4by4_save(np.reshape(test_origin,(-1,784)),file_name + '/1_ground_true.png') 
my_lib.mnist_4by4_save(np.reshape(test_ref,(-1,784)),file_name + '/1_input_ref.png')

        
r,val_e = sess.run([G_y,content_loss],feed_dict={u : np.reshape(test_input,(-1,28,28,1)),
    ref : np.reshape(test_ref,(-1,28,28,1)), t : np.reshape(test_origin,(-1,28,28,1)),
    isTrain : False})
  
my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/1_result.png')
 
print(val_e)


'''

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
    
    _ , content_e, mse_e, cross_entropy_e= sess.run([content_optim, content_loss,mse,cross_entropy],
        feed_dict={u : np.reshape(train_input,(-1,28,28,1)), ref : np.reshape(train_ref,(-1,28,28,1)),
        t : np.reshape(train_origin,(-1,28,28,1)),isTrain : True})
    content_error.append(content_e)
    mse_error.append(mse_e)
    cross_entropy_error.append(cross_entropy_e)

    if i%1000 == 0:
        print('%8d,content_e : %.6f, mse : %.6f, cross_entropy : %.6f'%(i, np.mean(content_error)
            ,np.mean(mse_error), np.mean(cross_entropy_error)))
        log_txt.write('%8d,content_e : %.6f, mse : %.6f, cross_entropy : %.6f\n'%(i, np.mean(content_error)
            ,np.mean(mse_error), np.mean(cross_entropy_error)))
        
        r = sess.run([G_y],feed_dict={u : np.reshape(test_input,(-1,28,28,1)),
            ref : np.reshape(test_ref,(-1,28,28,1)), t : np.reshape(test_origin,(-1,28,28,1)),
            isTrain : False})
  
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
'''






