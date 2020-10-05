import tensorflow as tf
import numpy as np

##Credit spectral norm layers: Miyato 2018
##Some modifications
def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm

def conv(x, channels, nchw=0, kernel=4, stride=2, padding='SAME', pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):

        if sn :
            data_format = 'NCHW' if nchw else 'NHWC'
            if not nchw:
                shape = [kernel, kernel, x.get_shape()[-1], channels]
                strides = [1, stride, stride, 1]
            else:
                shape = [kernel, kernel, x.get_shape()[1], channels]
                strides = [1, 1, stride, stride]

            w = tf.get_variable("kernel", shape=shape,
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), regularizer=None)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=strides, padding=padding, data_format=data_format)
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else :
            data_format = 'channels_first' if nchw else 'channels_last'
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=None,
                                 kernel_regularizer=None,
                                 strides=stride, use_bias=use_bias, padding=padding, data_format=data_format)
        return x

def deconv(x, channels, nchw=0, kernel=4, stride=2, use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        if sn :
            data_format = 'NCHW' if nchw else 'NHWC'
            if not nchw:
                x_shape = x.get_shape().as_list()
                output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]
                strides = [1, stride, stride, 1]
                kernel_shape = [kernel, kernel, x_shape[-1], channels]
            else:
                x_shape = x.get_shape().as_list()
                output_shape = [x_shape[0], channels, x_shape[2] * stride, x_shape[3] * stride]
                strides = [1, 1, stride, stride]
                kernel_shape = [kernel, kernel, x_shape[1], channels]

            w = tf.get_variable("kernel", shape=kernel_shape,
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), regularizer=None)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=strides, padding='SAME', data_format=data_format)
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else :
            data_format = 'channels_first' if nchw else 'channels_last'
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=None, kernel_regularizer=None,
                                           strides=stride, padding='SAME', use_bias=use_bias, data_format=data_format)
        return x

def fully_connected(x, units, use_bias=True, sn=False, scope='fully_0', kernel_initializer=None):
    with tf.variable_scope(scope):
        if sn :
            if kernel_initializer == None: kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
            x = tf.layers.flatten(x)
            shape = x.get_shape().as_list()
            channels = shape[-1]
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                     initializer=kernel_initializer, regularizer=None)
            if use_bias :
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))
                x = tf.matmul(x, spectral_norm(w)) + bias
            else :
                x = tf.matmul(x, spectral_norm(w))
        else :
            x = tf.layers.dense(x, units=units, kernel_initializer=kernel_initializer, kernel_regularizer=None, use_bias=use_bias)
        return x

#Credit self attention layers: Zhang 2018
#Some modifications
def attention(x, ch, sn=True, scope=''):
    batch_size, height, width, num_channels = x.get_shape().as_list()
    f = conv(x, ch / 8, kernel=1, stride=1, sn=sn, scope=scope+'_f_conv')  # [bs, h, w, c']
    #f = tf.layers.max_pooling2d(f, [2,2], 2)

    g = conv(x, ch / 8, kernel=1, stride=1, sn=sn, scope=scope+'_g_conv')  # [bs, h, w, c']

    h = conv(x, ch / 1, kernel=1, stride=1, sn=sn, scope=scope+'_h_conv')  # [bs, h, w, c]
    #h = tf.layers.max_pooling2d(h, [2,2], 2)
    # N = h * w
    #s = tf.matmul(tf.reshape(g, [batch_size,height*width,ch/8]), tf.reshape(f, [batch_size,height*width/4,ch/8]), transpose_b=True)  # # [bs, N, N]
    s = tf.matmul(tf.reshape(g, [batch_size,height*width,ch/8]), tf.reshape(f, [batch_size,height*width,ch/8]), transpose_b=True)

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, tf.reshape(h, [batch_size,height*width,ch]))  # [bs, N, C]
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=[batch_size, height, width, num_channels])  # [bs, h, w, C]
    o = conv(o, ch, kernel=1, stride=1, sn=sn, scope=scope+'_attn_conv')
    x = gamma * o + x

    return x

#Credit special layers: Karras 2018
#Some modifications
def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable('weight', shape=shape, initializer=init) * runtime_coef

def apply_bias(x, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])

def dense(x, fmaps, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

def conv2d(x, fmaps, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

def upscale2d_conv2d(x, fmaps, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

def conv2d_downscale2d(x, fmaps, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    # Fused => perform both ops simultaneously using tf.nn.conv2d().
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')
