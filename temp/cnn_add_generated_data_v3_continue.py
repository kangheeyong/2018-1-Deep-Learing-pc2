import tensorflow as tf
import numpy as np
import os
import sys
import time
import my_lib 
import time 

os.environ["CUDA_VISIBLE_DEVICES"]="3"



start = time.time()

  
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('_continue.py')[0]

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

   
sess = tf.InteractiveSession()


new_saver = tf.train.import_meta_graph(file_name + '/para.cktp.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(file_name + '/'))



cnn_u = sess.graph.get_tensor_by_name('cnn_u:0')
cnn_t = sess.graph.get_tensor_by_name('cnn_t:0')


cnn_loss = sess.graph.get_tensor_by_name('cnn_loss:0')
cnn_optim = sess.graph.get_operation_by_name('cnn_optim')

cnn_accuracy = sess.graph.get_tensor_by_name('cnn_accuracy:0')


test_input = np.reshape(mnist.test.images,(-1, 28, 28, 1))
test_label = mnist.test.labels


np.random.seed(int(time.time()))


log_txt = open(file_name +'/log.txt','w')


for i in range(5000000) :

  
    cnn_img, cnn_label = mnist.train.next_batch(50)
    cnn_img = np.reshape(cnn_img,(-1,28,28,1))
    cnn_label = np.reshape(cnn_label,(-1,10))

    cnn_z = np.round(np.random.normal(0,1,size=(50,1,1,100)),1)
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





