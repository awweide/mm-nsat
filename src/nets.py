import tensorflow as tf
import numpy as np

from ops import * #Custom network elements

def get_networks(a, IMAGE_SHAPE):
    IMAGE_H,IMAGE_W,IMAGE_C=IMAGE_SHAPE
    #DEFINE NETWORKS  
    _choice = [tf.nn.relu, None],[tf.nn.selu, tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')]
    d_act, d_ki = _choice[a.d_selu]
    g_act, g_ki = _choice[a.g_selu]

    #General ReLU-StridedConv network, supporting NCHW, all relevant resolutions and spectral normalization
    #Number of filters is high, but manageable for m_dim==32 at resolutions 32-1024
    def ConvNDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            #Strided convolutions from FULL RESOLUTION -> 2x2
            current_resolution = IMAGE_H
            while current_resolution > 2:
                f_out = int(a.m_dim * 128.0 / current_resolution)
                x = tf.nn.relu(conv(x, f_out, nchw=a.net_nchw, kernel=3, stride=2, sn=a.d_sn, scope='down_conv_'+str(current_resolution)))
                current_resolution /= 2
            #Final covolution from 2x2 -> logit D_l (this conv is equivalent to a fully connected layer)
            x = (conv(x, 1, nchw=a.net_nchw, kernel=2, stride=2, sn=False, scope='logit_down_conv'))
            return x
    def ConvNGen(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            #Latents to 4x4xCH base image
            x = tf.nn.relu(fully_connected(z, 4*4*8*a.m_dim))
            if not a.net_nchw: x = tf.reshape(x, [-1, 4, 4, 8*a.m_dim])
            else: x = tf.reshape(x, [-1, 8*a.m_dim, 4, 4])
            
            #Strided transpose convolutions from 4x4 -> FULL RESOLUTION
            current_resolution = 4
            while current_resolution < IMAGE_H:
                f_out = int(a.m_dim * 128.0 / current_resolution)
                x = tf.nn.relu(deconv(x, f_out, nchw=a.net_nchw, kernel=3, stride=2, sn=a.g_sn, scope='up_conv_'+str(current_resolution)))
                current_resolution *= 2
            #Final convolution from FULL RESOLUTION FILTERS to FULL RESOLUTION RGB: note kernel=1
            x = tf.nn.tanh(deconv(x, IMAGE_C , nchw=a.net_nchw, kernel=1, stride=1, sn=False, scope='channel_conv'))
            return x

    #DCGAN
    def DCGGen(z):
         with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            layers = int(np.log2(IMAGE_H))-2
            if not a.net_nchw: x = tf.reshape(z, [-1, 4, 4, 8])
            else: x = tf.reshape(z, [-1, 8, 4, 4])
            x = deconv(x,1024,nchw=a.net_nchw,kernel=4,stride=1,use_bias=not a.g_bn,sn=a.g_sn,scope='dc0')
            if a.g_bn: x =  tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            for i in xrange(1,layers):
                x = deconv(x,1024/2**i,nchw=a.net_nchw,kernel=4,stride=2,use_bias=not a.g_bn,sn=a.g_sn,scope='dc'+str(i))
                if a.g_bn: x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                if a.g_sa and (i==1 or i==3): x = attention(x, 1024/2**i, sn=True, scope='dc'+str(i))
            return tf.nn.tanh(deconv(x,IMAGE_C,nchw=a.net_nchw,kernel=4,stride=2,use_bias=True,sn=False,scope='dc_end'))
    def DCGDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            layers = int(np.log2(IMAGE_H))-2
            for i in xrange(0,layers):
                x = conv(x,2**(i+11-layers),nchw=a.net_nchw,kernel=4,stride=2,use_bias=False,sn=a.d_sn,scope='dc'+str(i))
                if a.d_bn: x = tf.layers.batch_normalization(x)
                x = tf.nn.leaky_relu(x)
                if a.d_sa and (i==layers-4 or i==layers-2): x = attention(x, 64*2**i, sn=True, scope='dc'+str(i))
            return tf.reshape(conv(x,1,nchw=a.net_nchw,kernel=4,stride=2,use_bias=True,sn=False,padding='VALID',scope='dc_end'), [-1,1])

    #32x32 Relu-StridedConv network: MNIST/CIFAR-10/MNIST-1K
    def Conv32Disc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(x, 1*a.m_dim, 3, 2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 2*a.m_dim, 3, 2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 4*a.m_dim, 3, 2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 8*a.m_dim, 3, 2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 1, 2, 2, padding='valid')
            x = tf.reshape(x, [-1, 1])
            return x
    def Conv32Gen(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(z, 4*4*4*a.m_dim, activation=tf.nn.relu)
            x = tf.reshape(x, [-1,4,4,4*a.m_dim])
            x = tf.layers.conv2d_transpose(x, 8*a.m_dim, 3, (2,2), padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, 4*a.m_dim, 3, (2,2), padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, 2*a.m_dim, 3, (2,2), padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, IMAGE_C, 1, (1,1), activation = tf.nn.tanh)
            return x

    #Fully connected network, supporting any number of layers
    def DenseDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            x = tf.reshape(x, [a.batch_size, np.prod(IMAGE_SHAPE)])
            for _ in xrange(a.d_layers-1):
                x = tf.layers.dense(x, a.m_dim, activation=d_act, kernel_initializer=d_ki)
            x = tf.layers.dense(x, 1)
            return x
    def DenseGen(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            for _ in xrange(a.g_layers-1):
                z = tf.layers.dense(z, a.m_dim, activation=g_act, kernel_initializer=g_ki)
            z = tf.layers.dense(z, IMAGE_H * IMAGE_W * IMAGE_C, activation = tf.nn.tanh)
            return tf.reshape(z, [-1, IMAGE_H, IMAGE_W, IMAGE_C])
    #Fully connected network, supporting any number of layers and spectral normalization
    def DenseSNDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            x = tf.reshape(x, [a.batch_size, np.prod(IMAGE_SHAPE)])
            for _ in xrange(a.d_layers-1):
                x = d_act(fully_connected(x, a.m_dim, sn=a.d_sn, kernel_initializer=d_ki, scope='fc_'+str(_)))
            x = fully_connected(x, 1, sn=a.d_sn, scope='fc_final')
            return x
    def DenseSNGen(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            for _ in xrange(a.g_layers-1):
                z = g_act(fully_connected(z, a.m_dim, sn=a.g_sn, kernel_initializer=g_ki, scope='fc_'+str(_)))
            z = tf.nn.tanh(fully_connected(z, IMAGE_H * IMAGE_W * IMAGE_C, sn=False, scope='fc_final'))
            return tf.reshape(z, [-1, IMAGE_H, IMAGE_W, IMAGE_C])

    #32x32 Relu-StridedConv network, supporting spectral normalization and extra, interlaced, unstrided layers
    def Conv32SNDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            for i in xrange((a.d_layers-1)/4): x = tf.nn.relu(conv(x, 1*a.m_dim, kernel=3, stride=1, sn=a.d_sn, scope='down_conv_1'+'_'+str(i)))
            x = tf.nn.relu(conv(x, 1*a.m_dim, kernel=3, stride=2, sn=a.d_sn, scope='down_conv_1'))
            for i in xrange((a.d_layers-2)/4): x = tf.nn.relu(conv(x, 2*a.m_dim, kernel=3, stride=1, sn=a.d_sn, scope='down_conv_2'+'_'+str(i)))
            x = tf.nn.relu(conv(x, 2*a.m_dim, kernel=3, stride=2, sn=a.d_sn, scope='down_conv_2'))
            for i in xrange((a.d_layers-3)/4): x = tf.nn.relu(conv(x, 4*a.m_dim, kernel=3, stride=1, sn=a.d_sn, scope='down_conv_3'+'_'+str(i)))
            x = tf.nn.relu(conv(x, 4*a.m_dim, kernel=3, stride=2, sn=a.d_sn, scope='down_conv_3'))
            for i in xrange((a.d_layers-4)/4): x = tf.nn.relu(conv(x, 8*a.m_dim, kernel=3, stride=1, sn=a.d_sn, scope='down_conv_4'+'_'+str(i)))
            x = tf.nn.relu(conv(x, 8*a.m_dim, kernel=3, stride=2, sn=a.d_sn, scope='down_conv_4'))
            x =           (conv(x, 1        , kernel=2, stride=2, sn=False, scope='down_conv_5'))
            return x
    def Conv32SNGen(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            x = tf.nn.relu(fully_connected(z, 4*4*4*a.m_dim))
            x = tf.reshape(x, [-1, 4, 4, 4*a.m_dim])
            for i in xrange((a.g_layers-1)/4): x = tf.nn.relu(deconv(x, 8*a.m_dim, kernel=3, stride=1, sn=a.g_sn, scope='up_conv_1'+'_' + str(i)))
            x = tf.nn.relu(deconv(x, 8*a.m_dim, kernel=3, stride=2, sn=a.g_sn, scope='up_conv_1'))
            for i in xrange((a.g_layers-2)/4): x = tf.nn.relu(deconv(x, 4*a.m_dim, kernel=3, stride=1, sn=a.g_sn, scope='up_conv_2'+'_' + str(i)))
            x = tf.nn.relu(deconv(x, 4*a.m_dim, kernel=3, stride=2, sn=a.g_sn, scope='up_conv_2'))
            for i in xrange((a.g_layers-3)/4): x = tf.nn.relu(deconv(x, 2*a.m_dim, kernel=3, stride=1, sn=a.g_sn, scope='up_conv_3'+'_' + str(i)))
            x = tf.nn.relu(deconv(x, 2*a.m_dim, kernel=3, stride=2, sn=a.g_sn, scope='up_conv_3'))
            for i in xrange((a.g_layers-4)/4): x = tf.nn.relu(deconv(x, 1*a.m_dim, kernel=3, stride=1, sn=a.g_sn, scope='ch_conv'+'_' + str(i)))
            x = tf.nn.tanh(deconv(x, IMAGE_C , kernel=1, stride=1, sn=False, scope='channel_conv'))
            return x

    #Residual DCGAN: experimental
    def ResDCGDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            layers = int(np.log2(IMAGE_H))-2
            if a.d_res: r = x
            for i in xrange(0,layers):
                if a.d_res: r = tf.keras.layers.MaxPooling2D()(r)
                x = conv(x,2**(i+11-layers),kernel=4,stride=2,use_bias=(i==0 or not a.d_bn),sn=a.d_sn,scope='dc'+str(i))
                if (i != 0 and a.d_bn): x = tf.layers.batch_normalization(x)
                x = tf.nn.leaky_relu(x)
                if a.d_sa and (i==layers-4 or i==layers-2): x = attention(x, 64*2**i, sn=True, scope='dc'+str(i))
                if a.d_res: x = tf.concat((x,r), axis=3)
            return tf.reshape(conv(x, 1, kernel=4,stride=2,use_bias=True,sn=False,padding='VALID',scope='dc_end'), [-1,1])
    def ResDCGGen(z):
         with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            layers = int(np.log2(IMAGE_H))-2
            x = tf.reshape(z, [-1,4,4,8])
            if a.g_res: r = x
            x = deconv(x,1024,kernel=4,stride=1,use_bias=not a.g_bn,sn=a.g_sn,scope='dc0')
            if a.g_bn: x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            for i in xrange(1,layers):
                if a.g_res: r = tf.keras.layers.UpSampling2D()(r)
                x = deconv(x,1024/2**i,kernel=4,stride=2,use_bias=not a.g_bn,sn=a.g_sn,scope='dc'+str(i))
                if a.g_bn: x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                if a.g_sa and (i==1 or i==3): x = attention(x, 1024/2**i, sn=True, scope='dc'+str(i))
                if a.g_res: x = tf.concat((x,r), axis=3)
            return tf.nn.tanh(deconv(x,IMAGE_C,kernel=4,stride=2,use_bias=True,sn=False,scope='dc_end'))
    
    #Batch normalized, residual fully-connected network, experimental
    def BNResDenseDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            x = tf.reshape(x, [a.batch_size, np.prod(IMAGE_SHAPE)])
            x = tf.layers.dense(x, a.m_dim, activation=d_act, kernel_initializer=d_ki)
            def ResBlock(x):
                s = x
                x = tf.layers.dense(x, a.m_dim, activation=d_act, kernel_initializer=d_ki)
                x = tf.layers.dense(x, a.m_dim, activation=d_act, kernel_initializer=d_ki)
                x = tf.layers.batch_normalization(x)
                return s+x
            for _ in xrange(a.d_layers-1):
                x = ResBlock(x)
            x = tf.layers.dense(x, 1)
            return x
    def BNResDenseGen(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            x = z
            x = tf.layers.dense(x, a.m_dim, activation=d_act, kernel_initializer=d_ki)
            def ResBlock(x):
                s = x
                x = tf.layers.dense(x, a.m_dim, activation=d_act, kernel_initializer=d_ki)
                x = tf.layers.dense(x, a.m_dim, activation=d_act, kernel_initializer=d_ki)
                x = tf.layers.batch_normalization(x)
                return s+x
            for _ in xrange(a.g_layers-1):
                x = ResBlock(x)
            x = tf.layers.dense(x, IMAGE_H * IMAGE_W * IMAGE_C, activation = tf.nn.tanh)
            return tf.reshape(x, [-1, IMAGE_H, IMAGE_W, IMAGE_C])

    #Specialized network: experimental
    def CifarGen(z):
        def ResUp(x, name):
            f_in = x.get_shape().as_list()[3]
            f_out = f_in
            x_res = deconv(x, f_out, kernel=2, stride=2, use_bias=True, sn=True, scope=name+'_res_up')
            if a.g_bn: x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = deconv(x,f_in, kernel=3, stride=1, use_bias=not a.g_bn, sn=a.g_sn, scope=name+'_same_conv')
            if a.g_bn: x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = deconv(x,f_out, kernel=3, stride=2, use_bias=not a.g_bn, sn=a.g_sn, scope=name+'_up_conv')
            x = tf.concat((x, x_res), axis=3)
            return x
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            x = tf.reshape(z, [-1,4,4,8])
            x = deconv(x,64,kernel=4,stride=1,use_bias=True, sn=a.g_sn,scope='filters_up')
            for i in xrange(0,3):
                x = ResUp(x, 'block'+str(i))
            return tf.nn.tanh(deconv(x,IMAGE_C,kernel=1,stride=1,use_bias=True,sn=False,scope='to_rgb'))
    def CifarDisc(x):
        def ResDown(x, name):
            #Num of filters
            f_in = x.get_shape().as_list()[3]
            f_out = f_in * 3
            #Residual
            x_res = conv(x, f_in, kernel = 2, stride = 2, use_bias=True, sn=True, scope=name+'_res_down')
            #Double conv block
            if a.d_bn: x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = conv(x,f_in, kernel=3, stride=1, use_bias=not a.d_bn, sn=a.d_sn, scope=name+'_same_conv')
            if a.d_bn: x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = conv(x,f_out, kernel=3, stride=2, use_bias=not a.d_bn, sn=a.d_sn, scope=name+'_up_conv')
            #Merge residual and block
            x = tf.concat((x, x_res), axis=3)
            return(x)
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            x = conv(x,12,kernel=1,stride=1,use_bias=True ,sn=a.d_sn,scope='from_rgb')
            for i in xrange(0,3):
                x = ResDown(x, 'block'+str(i))
            return tf.reshape(conv(x, 1, kernel=4,stride=2,use_bias=True,sn=False,padding='VALID',scope='to_logits'), [-1,1])

    #Nets implementing StyleGAN initialization with gain to correct variance
    #Experimental
    def ScaleConvGen(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            #4x4 -> full resolution        
            layers = int(np.log2(IMAGE_H))-2
            with tf.variable_scope('Latent_to_4x4'):
                x = tf.nn.relu(apply_bias(dense(z, 4*4*8*a.m_dim, use_wscale=True)))
            x = tf.reshape(x, [-1, 8*a.m_dim, 4, 4])

            for i in xrange(layers):        
                with tf.variable_scope('Upconv_' + str(i)):
                    x = tf.nn.relu(apply_bias(upscale2d_conv2d(x,int(a.m_dim*32.0/2**i), kernel=3, use_wscale=True)))
            with tf.variable_scope('Fullres_filters_to_image'):
                x = tf.nn.tanh(apply_bias(conv2d(x, IMAGE_C , kernel=1, gain=1, use_wscale=True)))
            #NCHW -> NHWC
            if not a.data_nchw: x = tf.transpose(x, [0,2,3,1])
            return x
    def ScaleConvDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            #NHWC -> NCHW
            if not a.data_nchw: x = tf.transpose(x, [0,3,1,2])

            #Full resolution -> 2x2
            layers = int(np.log2(IMAGE_H))-1
            for i in xrange(layers):        
                with tf.variable_scope('Downconv_' + str(i)):
                    x = tf.nn.relu(apply_bias(conv2d_downscale2d(x,int(a.m_dim*2**(i+6-layers)), kernel=3, use_wscale=True)))
            with tf.variable_scope('2x2_to_logit'):
                x = apply_bias(conv2d_downscale2d(x, 1, kernel=1, gain=1, use_wscale=True))
            return x
    def DenseDiscScaled(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            x = tf.reshape(x, [a.batch_size, np.prod(IMAGE_SHAPE)])
            for layer_idx in xrange(a.d_layers):
                with tf.variable_scope('Dense%d' % layer_idx):
                    [fmaps, act] = [1, tf.identity] if layer_idx == a.d_layers - 1 else [a.m_dim, tf.nn.relu]
                    x = dense(x, fmaps=fmaps, gain=np.sqrt(2), use_wscale=True, lrmul=1)
                    x = apply_bias(x, lrmul=1)
                    x = act(x)
            return x
    def DenseGenScaled(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            x = z
            for layer_idx in xrange(a.g_layers):
                with tf.variable_scope('Dense%d' % layer_idx):
                    [fmaps, act] = [np.prod(IMAGE_SHAPE), tf.nn.tanh] if layer_idx == a.g_layers - 1 else [a.m_dim, tf.nn.relu]
                    x = dense(x, fmaps=fmaps, gain=np.sqrt(2), use_wscale=True, lrmul=1)
                    x = apply_bias(x, lrmul=1)
                    x = act(x)
            return tf.reshape(x, [-1,] + IMAGE_SHAPE)


    dnet_dict = {'dense':DenseDisc,'sndense':DenseSNDisc,'conv32':Conv32Disc,'conv32sn':Conv32SNDisc,'convn':ConvNDisc,'dcg':DCGDisc}
    gnet_dict = {'dense':DenseGen,'sndense':DenseSNGen,'conv32':Conv32Gen,'conv32sn':Conv32SNGen,'convn':ConvNGen,'dcg':DCGGen}
    D,G = dnet_dict[a.d_net], gnet_dict[a.g_net]
    return D,G

