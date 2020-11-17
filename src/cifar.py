import numpy as np
import tensorflow as tf
import tensorflow
import pickle
from scipy import ndimage

EPOCHS = 200
BATCH_SIZE = 128

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = tensorflow.keras.layers.Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=tensorflow.keras.regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = tensorflow.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tensorflow.keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = tensorflow.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tensorflow.keras.layers.Activation(activation)(x)
        x = conv(x)
    return x

#Based on Keras tutorial from https://keras.io/examples/cifar10_resnet/
def resnet_v2(input_shape, depth, num_classes=10):
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = tensorflow.keras.layers.Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tensorflow.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.AveragePooling2D(pool_size=8)(x)
    y = tensorflow.keras.layers.Flatten()(x)
    outputs = tensorflow.keras.layers.Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def flip(x):
    x = tf.image.random_flip_left_right(x)
    return x
def color(x):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x
def rotate(x):
    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=10, dtype=tf.int32))
def zoom(x):
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]
    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]
    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))
# Add augmentations
augmentations = [flip, color, zoom, rotate]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
np_y_test = np.zeros((y_test.shape[0], 10), dtype=np.uint8)
np_y_test[np.arange(y_test.shape[0]), np.squeeze(y_test)] = 1
np_x_test = x_test/255.0 * 2.0 - 1.0

np_y_train = np.zeros((y_train.shape[0], 10), dtype=np.uint8)
np_y_train[np.arange(y_train.shape[0]), np.squeeze(y_train)] = 1
np_x_train = x_train/255.0 * 2.0 - 1.0

IMAGE_SHAPE = [32,32,3]

m_dim = 128
model = resnet_v2( (32,32,3), depth=29)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True)

model.fit_generator(datagen.flow(np_x_train, np_y_train, BATCH_SIZE), epochs=EPOCHS)
model.save('data/data/keras_resnet_cifar10_classifier.h5', include_optimizer=False)

#model = tf.keras.models.load_model('data/keras_resnet_cifar10_classifier.h5')
print model.evaluate(np_x_test, np_y_test, batch_size=BATCH_SIZE)
