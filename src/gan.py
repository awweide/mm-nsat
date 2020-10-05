## This file is mostly generic code for training GANs, aside from:
## Implementation of DJSCD, the class distribution divergence,
## gradient unit normalization and resetting of Adam momenta for the generator
## Network architectures in nets.py and cost functions in costs.py

#Library imports
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc as imageio
from scipy.signal import fftconvolve
from scipy.linalg import sqrtm
import os
import sys
import pickle
import time
import json
#FID SPECIFIC IMPORTS
from functools import partial
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
import tarfile

#Local imports
from nets import get_networks    #Network definitions
from costs import get_costs      #Cost definitions
from data import get_data        #Dataset handling

def main(a):
    #Create subdirectory for outputs from each run
    if not os.path.isdir(a.output_folder): os.mkdir(a.output_folder)
    else: raise ValueError('Invalid output_folder')

    #Save configuration settings to output folder
    with open(a.output_folder + '/args.json', 'w') as _: json.dump(vars(a), _)

    #Data handling defined in data.py: get_batch(data,i) returns a batch of real data
    #FFHQ is treated as a special case to optimize data flow: next_batch is a generator for FFHQ real data, preprocess transforms uint8[0,255]->float32[-1,1] 
    data, get_batch, N_SAMPLES, IMAGE_SHAPE, next_batch, preprocess = get_data(a)
    IMAGE_H,IMAGE_W,IMAGE_C= IMAGE_SHAPE

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=a.gpus
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    DEVICES = ['/gpu:{}'.format(i) for i in xrange(len(a.gpus.split(',')))] #GPU-naming

    #DEFINE TENSORFLOW GRAPH#
    D,G = get_networks(a, IMAGE_SHAPE)                                                  #Networks defined in nets.py
    z_feed = tf.placeholder(tf.float32, [a.batch_size, a.z_dim])                        #Generator noise inputs from feed_dict
    if 'ffhq' in a.dataset: x_feed = preprocess(next_batch)                             #FFHQ real data from tf generator
    else: x_feed = tf.placeholder(tf.float32, [a.batch_size, IMAGE_H, IMAGE_W, IMAGE_C])#Else real data from feed_dict

    ## GPU PARALLELIZATION ##
    x_gen, d_feed, d_gen, x_interp, d_interp = [],[],[],[],[]
    for device_index, (device, x_feed_, z_feed_) in enumerate(zip(DEVICES, tf.split(x_feed, len(DEVICES)), tf.split(z_feed, len(DEVICES)))):
        with tf.device(device), tf.name_scope('device_index'):
            x_gen_ = G(z_feed_)
            x_gen.append(x_gen_)
            d_feed.append(D(x_feed_))
            d_gen.append(D(x_gen_))
            #x_interp and d_interp: real-fake data interpolations only for WGAN-GP
            interp_eps_ = tf.random_uniform([a.batch_size/len(DEVICES),1,1,1], 0.0, 1.0)
    	    x_interp_ = (0.0 + interp_eps_) * x_feed_ + (1.0 - interp_eps_) * x_gen_
            x_interp.append(x_interp_)
            d_interp.append(D(x_interp_))
            #end interp
    x_gen, d_feed, d_gen, x_interp, d_interp = tf.concat(x_gen,axis=0),tf.concat(d_feed,axis=0),tf.concat(d_gen,axis=0),tf.concat(x_interp,axis=0),tf.concat(d_interp,axis=0)
    ## END GPU PARALLELIZATION ##

    #Compute diversities relative to dataset diversity: only for logging
    def batch_diversity(x0,x1):
        meansquare_distance = tf.reduce_mean(tf.square(x0 - x1[::-1]), axis=[1,2,3])
        return tf.reduce_mean(tf.sqrt(meansquare_distance))
    x_gen_div = batch_diversity(x_gen,x_gen)
    x_feed_div = batch_diversity(x_feed,x_feed)
    x_mix_div = batch_diversity(x_gen,x_feed)

    #Maintain exponential moving average of R rescaling factor - not in use
    #R_ema = tf.Variable(10.0, trainable=False)
    #R_ema_op = R_ema.assign(1.0-1.0/(a.g_cost_parameter+1e-18) * R_ema + 1.0/(a.g_cost_parameter+1e-18) * R)

    #d_feed, d_gen == Dl(x),Dl(G(z)), sig_d_feed, sig_d_gen == Dp(x),Dp(G(z))
    sig_d_feed, sig_d_gen = tf.sigmoid(d_feed), tf.sigmoid(d_gen)
    #costs defined in costs.py, pass all tensors which may be useful as arguments
    disc_loss, gen_loss = get_costs(a, x_gen, x_feed, d_gen, d_feed, sig_d_feed, sig_d_gen, d_interp, x_interp) 
    
    #Compute gradients and gradient norms explicitly
    #The steps from cost_func to training_op are more complicated than normal, in order to allow an explicit gradient normalization step
    gen_vars, disc_vars = tf.trainable_variables(scope='Gen'), tf.trainable_variables(scope='Disc')
    gen_grad, disc_grad = tf.gradients(gen_loss, gen_vars, colocate_gradients_with_ops=True), tf.gradients(disc_loss, disc_vars, colocate_gradients_with_ops=True)
    gen_grad_norm, disc_grad_norm = tf.global_norm(gen_grad), tf.global_norm(disc_grad)
                                                                           
    ## RENORMING OF GRADIENT ##
    norm_epsilon = 1e-18
    #Explicit MM-nsat renormalization: deprecated in favor of MM-nsat cost function with tf.stop_gradient rescaling
    def norm_frac(): return (1 + norm_epsilon - tf.reduce_mean(sig_d_gen)) / (tf.reduce_mean(sig_d_gen) + norm_epsilon)
    #Renormalize MM-GAN to exact NS-GAN norm, computationally expensive
    def norm_nsat(): return tf.global_norm(tf.gradients(msce_gen_1, gen_vars)) / (gen_grad_norm + norm_epsilon)
    #Unit normalization (to number of parameters)
    def norm_unit(): return 1. / (gen_grad_norm + norm_epsilon) * np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables('Gen')])
    def norm_none(): return tf.constant(1.0)
    norm_dict = {'frac':norm_frac,'nsat':norm_nsat,'unit':norm_unit,'none':norm_none}
    renorm = norm_dict[a.g_renorm]()
    gen_grad = map(lambda x: x * renorm, gen_grad)
    ## END RENORMING OF GRADIENT

    ## CHOICE OF OPTIMIZER ##
    if a.opt == 'adam':
        d_opt = tf.train.AdamOptimizer(learning_rate=a.d_lr, beta1=a.d_beta1, beta2=a.d_beta2, epsilon=a.d_adameps)
        g_opt = tf.train.AdamOptimizer(learning_rate=a.g_lr, beta1=a.g_beta1, beta2=a.g_beta2, epsilon=a.g_adameps)
    elif a.opt == 'sgd':
        d_opt = tf.train.GradientDescentOptimizer(learning_rate=a.d_lr)
        g_opt = tf.train.GradientDescentOptimizer(learning_rate=a.g_lr)
    else: raise ValueError('Invalid optimizer')

    ## OPTIMIZATION OPS ##
    disc_train_op = d_opt.apply_gradients(zip(disc_grad, disc_vars))
    gen_train_op = g_opt.apply_gradients(zip(gen_grad, gen_vars))
    gen_opt_mom_reset_op = tf.group( [tf.assign(v, tf.zeros_like(v)) for v in g_opt.variables() if 'Gen' in v.name] ) #Special op to allow reseting G's adam momenta
    #for v in g_opt.variables(): print v

    ## FRECHET INCEPTION DISTANCE ##
    def get_graph_def_custom(filename='inceptionv1_for_inception_score.pb',tar_filename='data/fid_inception_model/frozen_inception_v1_2015_12_05.tar.gz'):
        with tarfile.open(tar_filename, 'r:gz') as tar:
            proto_str = tar.extractfile(filename).read()
        return graph_pb2.GraphDef.FromString(proto_str)

    inception_images = tf.placeholder(tf.float32, [a.batch_size, 3, None, None])
    activations1 = tf.placeholder(tf.float32, [None, None], name = 'activations1')
    activations2 = tf.placeholder(tf.float32, [None, None], name = 'activations2')
    fcd = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(activations1, activations2)

    def inception_activations(images=inception_images, num_splits = 1):
        images = tf.transpose(images, [0, 2, 3, 1])
        size = 299
        images = tf.image.resize_bilinear(images, [size, size])
        generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
        activations = functional_ops.map_fn(fn = partial(tf.contrib.gan.eval.run_inception,default_graph_def_fn=get_graph_def_custom,output_tensor = 'pool_3:0'),
            elems = array_ops.stack(generated_images_list),parallel_iterations = 1,back_prop = False,swap_memory = True,name = 'RunClassifier')
        activations = array_ops.concat(array_ops.unstack(activations), 0)
        return activations
    if not a.eval_skip: activations=inception_activations()

    def get_inception_activations(inps):
        n_batches = inps.shape[0]//a.batch_size
        act = np.zeros([n_batches * a.batch_size, 2048], dtype = np.float32)
        for i in range(n_batches):
            inp = inps[i * a.batch_size:(i + 1) * a.batch_size] / 255. * 2 - 1
            act[i * a.batch_size:(i + 1) * a.batch_size] = activations.eval(feed_dict = {inception_images: inp})
        return act
    #STYLEGAN FID IMPLEMENTATION
    def acts2stats(acts):
        mu = np.mean(acts, axis=0)
        sigma = np.cov(acts, rowvar=False)
        return mu,sigma
    def stats2dist(mu_real, sigma_real, mu_fake, sigma_fake):
        m = np.square(mu_fake - mu_real).sum()
        s, _ = sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        return np.real(dist)
            
    #RESCALE, CAST AND TRANSPOSE SAMPLES TO CONFORM TO INCEPTION NETWORK EXPECTATIONS
    def cast_inception(im):
        #[-1,1] float32 -> [0,255] uint8
        im = np.array(im, copy=True)
        im = ((im+1)*(256/2)).astype(np.uint8)
        #IF SINGLE CHANNEL MNIST IMAGE; TRIPLICATE CHANNEL TO FAKE RGB
        if a.dataset == 'mnist': im = np.repeat(im,3,axis=3)
        #INCEPTION EXPECTS CHANNELS FIRST
        if not a.net_nchw: im = np.transpose(im, [0,3,1,2])
        return im
    ## END FRECHET DISTANCE ##
    ## END DEFINE TENSORFLOW GRAPH ##

    ## METRICS ##
    def initialize_class_distribution():
        if a.dataset == 'mnist' or a.dataset == 'cifar': class_counts_gen = np.zeros(10)
        elif a.dataset == 'mnist1k': class_counts_gen = np.zeros(1000)
        elif 'cats' in a.dataset or 'ffhq' in a.dataset: class_counts_gen = np.zeros(1)
        else: raise ValueError('Classifier not implemented for dataset')
        return class_counts_gen

    def update_class_distribution(x_gen, class_counts, classifier):
        if a.dataset == 'mnist1k':
            x_gen_onehots_0 = np.argmax(classifier.predict(x_gen[:,:,:,0:1], batch_size=x_gen.shape[0]), axis=1)
            x_gen_onehots_1 = np.argmax(classifier.predict(x_gen[:,:,:,1:2], batch_size=x_gen.shape[0]), axis=1)
            x_gen_onehots_2 = np.argmax(classifier.predict(x_gen[:,:,:,2:], batch_size=x_gen.shape[0]), axis=1)
            x_gen_onehots = x_gen_onehots_0 + 10*x_gen_onehots_1 + 100*x_gen_onehots_2
            for i in xrange(x_gen.shape[0]): class_counts[x_gen_onehots[i]] += 1
        elif a.dataset == 'mnist' or a.dataset == 'cifar':
            x_gen_onehots = classifier.predict(x_gen, batch_size=x_gen.shape[0])
            x_gen_classes = np.argmax(x_gen_onehots, axis=1)
            for cls in x_gen_classes: class_counts[cls] += 1
        elif 'cats' in a.dataset or 'ffhq' in a.dataset:
            class_counts[0] += len(x_gen)
        else: raise ValueError('Classifier not implemented for dataset')
        return class_counts

    def print_class_distribution(class_freqs, js):
        print 'Fake classes: %.4f' % js
        if a.dataset != 'mnist1k':
            for i,v in enumerate(class_freqs): print "%d: %.2f " % (i, v),
        print ''

    #Jensen-Shannon distance between class distributions of fake and real data (bounded [0,1])
    #Real data defaults to even class balance
    #Somewhat awkward implementation to give correct beavior for exactly 0 class frequency
    def get_jensen_shannon(f, r=None):
        if a.dataset == 'mnist' or a.dataset == 'cifar':
            if r == None: r = np.zeros(10)+0.1
            m = 0.5*(f+r)
            s = np.sum([fi*np.log(fi/mi) for fi,mi in zip(f,m) if fi>0]) + np.sum([ri*np.log(ri/mi) for ri,mi in zip(r,m) if ri>0])
            return 0.5 * s / np.log(2)
        elif a.dataset == 'mnist1k':
            if r == None: r = np.zeros(1000)+0.001
            m = 0.5*(f+r)
            s = np.sum([fi*np.log(fi/mi) for fi,mi in zip(f,m) if fi>0]) + np.sum([ri*np.log(ri/mi) for ri,mi in zip(r,m) if ri>0])
            return 0.5 * s / np.log(2)
        #NO CLASS LABELS OR CLASSIFIER FOR CATS AND FFHQ
        elif 'cats' in a.dataset or 'ffhq' in a.dataset:
            return 1.0
        else: raise ValueError('Jensen-Shannon not implemented for dataset')

    def eval_metrics_acts(fid_batches, fid_mu_real, fid_sigma_real, diversity, epoch, session):
        start_time = time.time()
        class_counts_gen = initialize_class_distribution()
        
        #REAL ACTIVATIONS
        if np.sum(fid_mu_real[0]) == 0: #COMPUTE ONLY IF ZERO INITIALIZED
            fid_acts_real = np.zeros(shape=[fid_batches*a.batch_size, 2048], dtype=np.float32)
            for j in xrange(fid_batches):            
                if 'ffhq' in a.dataset: _x_real = session.run(next_batch) #PREPROCESS CANCELS AGAINST CAST_INCEPTION FOR FFHQ
                else: _x_real = cast_inception(get_batch(data,j))
                fid_acts_real[j*a.batch_size:(j+1)*a.batch_size,:] = get_inception_activations(_x_real)
            fid_mu_real[:], fid_sigma_real[:] = acts2stats(fid_acts_real)
        #GENERATED ACTIVATIONS
        fid_acts_gen  = np.zeros(shape=[fid_batches*a.batch_size, 2048], dtype=np.float32)
        for j in xrange(fid_batches):
            _x_gen = session.run(x_gen,feed_dict={z_feed:np.random.normal(size=[a.batch_size,a.z_dim])})
            fid_acts_gen[j*a.batch_size:(j+1)*a.batch_size,:] = get_inception_activations(cast_inception(_x_gen))
            class_counts_gen = update_class_distribution(_x_gen, class_counts_gen, clf)
        fid_mu_gen, fid_sigma_gen = acts2stats(fid_acts_gen)
        fid_value = stats2dist(fid_mu_real, fid_sigma_real, fid_mu_gen, fid_sigma_gen)

        class_frequencies_gen = class_counts_gen*(1.0/np.sum(class_counts_gen))
        jensen_shannon_gen = get_jensen_shannon(class_frequencies_gen)
        print_class_distribution(class_frequencies_gen, jensen_shannon_gen)
        print 'FID value: ', fid_value, '\n'
        with open(a.output_folder + '/fid_'+str(int(epoch))+'.txt','w') as f: f.write(str(fid_value))
        with open(a.output_folder + '/cls_'+str(int(epoch))+'.txt','w') as f: f.write(str(class_frequencies_gen) + '\n' + str(jensen_shannon_gen))
        with open(a.output_folder + '/div_'+str(int(epoch))+'.txt','w') as f: f.write(str(diversity))
        print('Evluation time: %f s' % (time.time() - start_time))
    ## END METRICS ##

    ## UTILITIES ##
    def confusion(real,fake): return np.mean( (np.sort(real) - np.sort(fake)[::-1]) <= 0 )
    def epoch_from_iter(i): return (i * a.batch_size * 1.0) / N_SAMPLES
    def mosaic_batch(images):
        #If net and data both in NCHW format, we only need NCHW->NHCW to save generated samples
        if a.net_nchw and a.data_nchw: images = np.transpose(images, [0,2,3,1])
        w = IMAGE_H #Assumes square images
        d = int(np.sqrt(a.batch_size)) #Largest possible square grid
        mosaic = np.zeros([w*d, w*d, 3])
        for u in xrange(d):
            for v in xrange(d):
                mosaic[w*u:w*(u+1), w*v:w*(v+1), :] = images[d*u+v,:,:,:]
        return mosaic

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        session.run(tf.global_variables_initializer())

        ##KERAS CLASSIFIER FOR CLASS DISTRIBUTIONS##
        if a.eval_skip or 'cats' in a.dataset or 'ffhq' in a.dataset: clf = None
        elif a.dataset == 'mnist' or a.dataset == 'mnist1k': clf = tf.keras.models.load_model('data/keras_mnist_classifier.h5')
        elif a.dataset == 'cifar': clf = tf.keras.models.load_model('data/keras_resnet_cifar10_classifier.h5')
        else: raise ValueError('Classifier not implemented for dataset')

        gen_vars_num, disc_vars_num = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables('Gen')]), np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables('Disc')])
        print 'Generator parameters:     ', gen_vars_num
        print 'Discriminator parameters: ', disc_vars_num

        #Fixed generator input to reuse when saving batches of generated samples during training
        z_fixed = np.random.normal(size=[a.batch_size,a.z_dim])
        #Initialize logging arrays - not pretty code
        ITERATIONS = (a.epochs * N_SAMPLES / a.batch_size) + 1
        log_xvals = np.arange(ITERATIONS)
        log_confusion = np.zeros(ITERATIONS); log_d_real = np.zeros(ITERATIONS); log_d_gen = np.zeros(ITERATIONS)
        log_lr_ratio = np.zeros(ITERATIONS); log_renorm = np.zeros(ITERATIONS); log_gen_grad_norm = np.zeros(ITERATIONS); log_disc_grad_norm = np.zeros(ITERATIONS)
        log_xg_div = np.zeros(ITERATIONS); log_xr_div = np.zeros(ITERATIONS); log_xm_div = np.zeros(ITERATIONS)
        #1D Convolution filter: only for smoothing plots of logged values
        c_size = np.minimum(a.epochs*N_SAMPLES/(a.batch_size*a.eval_n),2500)
        c_filter = np.ones(ITERATIONS/c_size)*((0.0+c_size)/ITERATIONS)
        def conv_smooth(arr,f):
            if a.eval_n > a.epochs: return arr #Make sure smoothing filter is not too large
            return fftconvolve(np.pad(arr, (0,len(f)-1), 'edge'), f, 'valid')

        fid_batches = (np.ceil(float(a.fid_n)/a.batch_size)).astype(int)
        #CACHE REAL DATA STATISTICS TO AVOID RECOMPUTING
        fid_mu_real = np.zeros(shape=[2048], dtype=np.float32)
        fid_sigma_real = np.zeros(shape=[2048,2048], dtype=np.float32)

        ## TRAINING LOOP ##
        for i in xrange(ITERATIONS):
            #Special case: reset generator Adam momenta
            if a.g_adamreset != 0:
                if (i % a.g_adamreset) == 0: _ = session.run(gen_opt_mom_reset_op)

            ## EVALUATION STEP ##
            if ((i % (ITERATIONS/a.eval_n)) == 0 or (i == (ITERATIONS-1))):
                if 'ffhq' not in a.dataset: feed_dict = {x_feed : get_batch(data,i), z_feed : np.random.normal(size=[a.batch_size,a.z_dim])}
                else: feed_dict = feed_dict={z_feed : np.random.normal(size=[a.batch_size,a.z_dim])}
                #Evaluate tensors
                _x_gen, _d_real, _d_gen, _x_gen_div, _x_real_div = session.run((x_gen, sig_d_feed, sig_d_gen, x_gen_div, x_feed_div), feed_dict=feed_dict)
       
                #Print cryptic information during training
                print epoch_from_iter(i), a.g_cost, a.g_renorm, a.g_cost_parameter, confusion(_d_real, _d_gen), _x_gen_div / _x_real_div
                #Save generated samples
                imageio.imsave(a.output_folder + '/gen_{}.png'.format(i), mosaic_batch(_x_gen))
                #Metrics: must be skipped to run without proper setup for FID and DJSCD
                if not a.eval_skip: eval_metrics_acts(fid_batches, fid_mu_real, fid_sigma_real, _x_gen_div/_x_real_div, epoch_from_iter(i), session)

                #Plot logged values - not pretty code
                fig, ax_log = plt.subplots()
                ax_log.set_yscale('log')
                ax_lin = ax_log.twinx()
                for arr,lab,ax,c in zip(
                    [log_confusion, log_d_real - log_d_gen, log_lr_ratio, log_gen_grad_norm/gen_vars_num, log_disc_grad_norm/disc_vars_num, log_xg_div/(log_xr_div+1e-12)],
                    ['$D(x)$:$D(G(z))$ overlap', '$D(x)$:$D(G(z))$ distance', '$R$ scaling factor', '$|\\nabla J_{G}|$ / #$\\theta$', '$|\\nabla J_{D}|$ / #$\phi$', '$G$ diversity'],
                    [ax_lin, ax_log, ax_log, ax_log, ax_log, ax_lin],
                    ['b','g','r','c','m','y']
                    ):
                    if ax==ax_lin:
                        ax.plot(epoch_from_iter(log_xvals), conv_smooth(np.clip(arr,0.0,1.0),c_filter), label=lab, color=c, alpha=0.5)
                    else:
                        ax.plot(epoch_from_iter(log_xvals), np.exp(conv_smooth(np.log(np.clip(arr,1e-9,1e6)),c_filter)), label=lab, color=c, alpha=0.5)
  
                ax_log.set_ylim((1e-9*0.80,1e6*1.25))
                ax_lin.set_ylim((-0.25,1.25))
                ax_log.set_xlim((0,a.epochs))
                ax_log.set_xlabel('Training epochs (batch size=' + str(a.batch_size) + ', #samples=' + str(N_SAMPLES) + ')')
                ax_lin.legend(loc='upper right',fancybox=True, framealpha=0.5)
                leg = ax_log.legend(loc='upper left',fancybox=True, framealpha=0.5)
                leg.remove()
                ax_lin.add_artist(leg)
                plt.title('Training diagnostics (clipped and smoothed)')
                plt.savefig(a.output_folder + '/all_pretty.jpg')
                plt.close()
                #Special case: dump pickle of values for early gradient plots
                if False: pickle.dump([log_lr_ratio, log_gen_grad_norm/gen_vars_num, log_disc_grad_norm/disc_vars_num], open(a.output_folder + '/gradients.pkl', 'wb'))
            ## END EVALUATION LOOP

            ## TRAINING STEP ##
            #Special feed_dict due to data handling for FFHQ
            if 'ffhq' not in a.dataset: feed_dict = {x_feed : get_batch(data,i), z_feed : np.random.normal(size=[a.batch_size,a.z_dim])}
            else: feed_dict = feed_dict={z_feed : np.random.normal(size=[a.batch_size,a.z_dim])}
            _, _, _x_gen, _d_real,_d_gen, _sig_d_real, _sig_d_gen, _x_gen_div, _x_real_div, _x_mix_div, _renorm, _gen_grad_norm, _disc_grad_norm = session.run(
            (disc_train_op, gen_train_op, x_gen, d_feed, d_gen, sig_d_feed, sig_d_gen, x_gen_div, x_feed_div, x_mix_div, renorm, gen_grad_norm, disc_grad_norm),
            feed_dict=feed_dict)
            #Logging
            log_confusion[i] = confusion(_d_real, _d_gen); log_renorm[i] = _renorm
            log_d_real[i] = np.mean(_d_real); log_d_gen[i] = np.mean(_d_gen)
            log_lr_ratio[i] = (1 + 1e-8 - np.mean(_sig_d_gen))/(np.mean(_sig_d_gen) + 1e-8)
            log_gen_grad_norm[i] = _gen_grad_norm; log_disc_grad_norm[i] = _disc_grad_norm
            log_xg_div[i] = _x_gen_div; log_xr_div[i] = _x_real_div; log_xm_div[i] = _x_mix_div
            ## END TRAINING LOOP

    #RESET FOR MULTIPLE RUNS FROM RUN.PY
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
