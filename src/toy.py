import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import scipy.misc
import sys
import pickle

try: cost = int(sys.argv[1])
except:
    print 'Erroneous cost, reverting to default value'
    cost = 0
try: a = float(sys.argv[2])
except:
    print 'Erroneous parameter a, reverting to default value'
    a = 0.0
try: data = sys.argv[3]
except:
    print 'Erroneous data set, reverting to default ring'
    data = 'ring'
try: folder = sys.argv[4]
except: folder = 'toy_default'

os.mkdir('out/' + folder)
path = 'out/' + folder + '/'

BATCH_SIZE = 2**7
X_DIM = 2
Z_DIM = 128
ITERATIONS = 10**5
plot_batches = 16

learning_rate = 1e-4

N_GPUS = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=''
DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)] #GPU-naming


if data == 'ring':
    modes = [ ((np.cos(np.pi*x), np.sin(np.pi*x)),(0.03,0.03)) for x in np.linspace(0,2,8, endpoint=False)]
    weights = np.ones(len(modes))
elif data == 'spiral':
    modes = [ ((0.3*x*np.cos(np.pi*x), 0.3*x*np.sin(np.pi*x)),(0.1,0.1)) for x in np.linspace(0,4.788,12, endpoint=True)]
    weights = np.linspace(1,8,len(modes), endpoint=False)
    ITERATIONS *= 10
else:
    raise ValueError('Dataset not implemented')

#Normalize weights
weights = weights / np.sum(weights)

def x_real_server():
    m = np.random.choice(range(len(weights)), size=BATCH_SIZE, replace=True, p=weights)    
    a = np.array( [ np.random.normal(*modes[_]) for _ in m] )
    return a

#x_fixed = np.concatenate([x_real_server() for _ in xrange(1024)])
def x_real_server_fixed():
    c = np.random.choice(range(len(x_fixed)), size=BATCH_SIZE, replace=False)
    return np.array( [x_fixed[_] for _ in c])

def z_rnd_server(): return np.random.normal(size=[BATCH_SIZE,Z_DIM])

def Disc(x,tr,name='Discriminator'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(x,Z_DIM)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x,Z_DIM)
        x = tf.nn.relu(x)
        return tf.layers.dense(x,1)

def Gen(z,tr,name='Generator'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        z = tf.layers.dense(z,Z_DIM)
        z = tf.nn.relu(z)
        z = tf.layers.dense(z,Z_DIM)
        z = tf.nn.relu(z)
        z = tf.layers.dense(z,X_DIM)
        return z

def modal_distance(x,m):
    u,o = np.array(m[0]),np.array(m[1])
    d = np.sum(np.absolute((x - u) / o),axis=1)
    return d / len(o)
def modality(x):
    for i,mode in enumerate(modes):
        if i==0:
            d = modal_distance(x,mode)
            m = np.zeros_like(d)
        else:
            d_new = modal_distance(x,mode)
            b = np.argmin((d,d_new), axis=0 )
            
            d = d + ((d_new-d)*b)
            m = m + ((i-m) * b)
    return m,d

def confusion(real,fake): return np.mean( (np.sort(real) - np.sort(fake)[::-1]) < 0 )
def probit_from_logit(logit): return 1. / (1. + np.exp(-logit))

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    x_feed = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, X_DIM])
    z_feed = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, Z_DIM])
    tr = tf.placeholder(dtype=tf.bool)
    x_gen = Gen(z_feed,tr)
    d_feed, d_gen = Disc(x_feed,tr), Disc(x_gen,tr)

    cost_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_feed), logits=d_feed))
    cost_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(d_gen), logits=d_gen))
    d_cost = cost_fake + cost_real

    sig_d_gen = tf.sigmoid(d_gen)
    R = (1.0 + 1e-18 - tf.reduce_mean(sig_d_gen)) / (tf.reduce_mean(sig_d_gen) + 1e-18)
    def g_cost_nsmmnsat(a): return (1.0 - a) * tf.nn.softplus(-d_gen) + a * -tf.nn.softplus(d_gen)*tf.stop_gradient(R)
    def g_cost_nsmm(a): return (1.0 - a) * tf.nn.softplus(-d_gen) + a * -tf.nn.softplus(d_gen)
    
    if cost==0:
        g_cost = g_cost_nsmm(a)
        info = 'Cost: ' + str(1-a) + ' * NS + ' +str(a) + ' *  MM'
    else:
        g_cost = g_cost_nsmmnsat(a)
        info = 'Cost: ' + str(1-a) + ' * NS + ' +str(a) + ' *  MM-NSAT'

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([_ for  _ in update_ops if 'Generator' in _.name]):
        gen_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
g_cost, var_list=tf.trainable_variables(scope='Generator'), colocate_gradients_with_ops=True)
    with tf.control_dependencies([_ for  _ in update_ops if 'Generator' in _.name]):
        gen_train_op_alt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
g_cost_nsmm(0.0), var_list=tf.trainable_variables(scope='Generator'), colocate_gradients_with_ops=True)
    with tf.control_dependencies([_ for  _ in update_ops if 'Discriminator' in _.name]):
        disc_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
d_cost, var_list=tf.trainable_variables(scope='Discriminator'), colocate_gradients_with_ops=True)

    session.run(tf.initialize_all_variables())

    #Train loop
    for i in xrange(1,ITERATIONS+1):
        _,_ = session.run( (gen_train_op, disc_train_op),  feed_dict={x_feed:x_real_server(), z_feed:z_rnd_server(), tr:True})

        #Output evaluation statistics and images
        if i%(ITERATIONS/10) == 0:
            tupl = [session.run( (x_gen, d_gen), feed_dict={z_feed:z_rnd_server(), tr:False}) for _ in xrange(plot_batches)]
            fake = np.concatenate( [_[0] for _ in tupl], axis=0)
            d_fake = np.concatenate( [_[1] for _ in tupl], axis=1)
            real = np.concatenate( [x_real_server() for _ in xrange(plot_batches) ], axis=0 )

            print ''
            print info + ', ITERATION: ' + str(i)
            print 'Mode # | Real freq | Gen freq'
            m,d = modality(fake)
            for index,_ in enumerate(modes):
                _d = d[m==index]
                _d = _d[_d<3.0]
                print index, weights[index], len(_d) / (1.0*plot_batches*BATCH_SIZE)
            print -1, 0, len(d[d>=3.0]) / (1.0*plot_batches*BATCH_SIZE)

            plt.close()
            plt.scatter(real[:,0], real[:,1], c='b', label='Real samples')
            plt.scatter(fake[:,0], fake[:,1], c='r', label='Fake samples')
            plt.xlim(-1.5,1.5)
            plt.ylim(-1.5,1.5)
            plt.legend()
            plt.title('Real and generated samples for toy data')
            plt.savefig(path+'im_{}.png'.format(i))
            plt.close()

    #Save samples at end of run
    pickle.dump([real, fake], open(path+'tuple_samples.pkl', 'wb'))
