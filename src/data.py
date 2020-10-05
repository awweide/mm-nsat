import tensorflow as tf
import numpy as np
import pickle
import os
import scipy.misc as imageio

def get_data(a):
    if a.dataset == 'mnist':
        N_SAMPLES = 60000
        IMAGE_SHAPE = [32,32,1]
        from tensorflow.examples.tutorials.mnist import input_data
        (mnist_images, _),(_,_) = tf.keras.datasets.mnist.load_data()
        mnist_images = np.reshape(mnist_images, [60000, 28, 28, 1])
        mnist_images = np.pad(mnist_images, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        mnist_images = mnist_images.astype(np.float32)
        data = mnist_images

    elif a.dataset == 'mnist1k':
        N_SAMPLES = 60000
        IMAGE_SHAPE = [32,32,3]
        from tensorflow.examples.tutorials.mnist import input_data
        (mnist_images, _),(_,_) = tf.keras.datasets.mnist.load_data()
        mnist_images = np.reshape(mnist_images, [60000, 28, 28, 1])
        mnist_images = np.pad(mnist_images, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        mnist_images = mnist_images.astype(np.float32)
        p0 = np.arange(N_SAMPLES)
        p1, p2 = np.random.permutation(N_SAMPLES), np.random.permutation(N_SAMPLES)
        mnist1k_images = np.array([np.concatenate( (mnist_images[p0[i]], mnist_images[p1[i]], mnist_images[p2[i]]), axis=2) for i in xrange(N_SAMPLES) ])
        data = mnist1k_images

    elif a.dataset == 'cifar':
        N_SAMPLES = 50000
        IMAGE_SHAPE = [32,32,3]
        def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
            with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
                batch = pickle.load(file)
            features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            labels = batch['labels']
            return features
        cifar10_images = np.vstack( [load_cfar10_batch('data/cifar-10', i) for i in xrange(1,6)] )
        data = cifar10_images

    elif 'cats' in a.dataset:
        #a.dataset == cats256 -> 256x256 resolution
        if '256' in a.dataset: 
            DATA_PATH = '../data/cats-data/cats_bigger_than_256x256'
            IMAGE_SHAPE = [256,256,3]
        elif '128' in a.dataset:
        #a.dataset == cats -> 128x128 resolution
            DATA_PATH = '../data/cats-data/cats_bigger_than_128x128'
            IMAGE_SHAPE = [128,128,3]
        else: raise ValueError('Cats resolution misspecified')

        cats_images = np.stack([(imageio.imread(DATA_PATH+'/'+im)) for im in os.listdir(DATA_PATH) if im.endswith('.jpg')], axis=0)
        N_SAMPLES =  cats_images.shape[0]
        data = cats_images

    elif 'ffhq' in a.dataset:
    	#Uses multi-resolution tfrecords as for StyleGAN
        resolution = int(a.dataset[4:]) #Fragile extraction of resolution
        subpath = str(int(np.log2(resolution))) #Hard coding for TFRecords names
        if len(subpath) < 2: subpath = '0' + subpath
        DATA_PATH = '../data/tensorflow_datasets/ffhq/ffhq-r' + subpath + '.tfrecords'
        IMAGE_SHAPE = [resolution,resolution,3]
        N_SAMPLES = 69940
        #StyleGAN parsing of tfrecords
        def parse_tfrecord_tf(record):
            features = tf.parse_single_example(record, features={
                'shape': tf.FixedLenFeature([3], tf.int64),
                'data': tf.FixedLenFeature([], tf.string)})
            data = tf.decode_raw(features['data'], tf.uint8)
            return tf.reshape(data, features['shape'])

        #Images in dataset are all the same shape, but do not have explicit shape assigned
        #TODO: May not be necessary with current implementation?
        def set_shape(image):
            image.set_shape([3,resolution,resolution])
            return image
        ds = tf.data.TFRecordDataset([DATA_PATH])
        ds = ds.map(parse_tfrecord_tf,2)
        ds = ds.map(set_shape)
        ds = ds.shuffle(2048)
        ds = ds.repeat(a.epochs*4) #Generator can be exhausted if used too often outside the training loop, for instance for FID evaluation
        ds = ds.batch(a.batch_size)
        ds_iterator = ds.make_one_shot_iterator()
        next_batch = ds_iterator.get_next()
        def preprocess(batch):
            #Transpose to NHWC unless network is NCHW : very inefficient
            if not a.net_nchw: batch = tf.transpose(batch, [0,2,3,1])
            #uint8[0,255] -> float32[-1,1]
            batch = tf.cast(batch, dtype=tf.float32)
            batch = batch * (2./255.) - 1
            return batch
        def get_batch(i): raise ValueError('FFHQ dataset does not implement get_batch')
        data = None

    else: raise ValueError('Invalid dataset')

    if 'ffhq' not in a.dataset:
        next_batch, preprocess = None, None #Only used by FFHQ
        def get_batch(data,i):
            batch_index = i % ((N_SAMPLES / a.batch_size) - 1)
            if batch_index == 0: np.random.shuffle(data)
            batch = data[batch_index*a.batch_size:(batch_index+1)*a.batch_size,:,:,:]
            #uint8[0,255] -> float32[-1,1]
            return 2.*batch/255. - 1

    return data, get_batch, N_SAMPLES, IMAGE_SHAPE, next_batch, preprocess
