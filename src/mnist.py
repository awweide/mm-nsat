import numpy as np
import tensorflow as tf
from scipy import ndimage

N_SAMPLES = 60000
EPOCHS = 5
IMAGE_SHAPE = [32,32,1]
BATCH_SIZE = 64

from tensorflow.examples.tutorials.mnist import input_data
mnist_images = input_data.read_data_sets("data/mnist/", one_hot=True)
mnist_images, mnist_labels = mnist_images.train.next_batch(N_SAMPLES)
mnist_images = np.reshape(mnist_images, [N_SAMPLES, 28, 28, 1])
mnist_images = np.pad(mnist_images, ((0,0),(2,2),(2,2),(0,0)), 'constant') 
mnist_images = mnist_images * 2 - 1.0

def expand_training_data(images, labels):
    expanded_images = []
    expanded_labels = []

    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        if j%100==0:
            print ('expanding data : %03d / %03d' % (j,np.size(images,0)))

        # register original data
        expanded_images.append(x[:,:,0])
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value 
        bg_value = -1.0 # this is regarded as background's value        
        image = x[:,:,0]

        for i in range(4):
            # rotate the image with random degree
            angle = np.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            # register new training data
            expanded_images.append(new_img_)
            expanded_labels.append(y)

    return np.array(expanded_images), np.array(expanded_labels)

augmented_mnist_images, augmented_mnist_labels = expand_training_data(mnist_images, mnist_labels)
#augmented_mnist_images, augmented_mnist_labels = expand_training_data(mnist_images[:200], mnist_labels[:200])
augmented_mnist_images = np.reshape(augmented_mnist_images, augmented_mnist_images.shape + (1,))
N_SAMPLES = augmented_mnist_images.shape[0]

def get_batch(i):
    batch_index = i % ((N_SAMPLES / BATCH_SIZE) - 1)
    if batch_index == 0:
        seed = np.random.get_state()
        np.random.shuffle(augmented_mnist_images)
        np.random.set_state(seed)
        np.random.shuffle(augmented_mnist_labels)
    return augmented_mnist_images[batch_index*BATCH_SIZE:(batch_index+1)*BATCH_SIZE,:,:,:], augmented_mnist_labels[batch_index*BATCH_SIZE:(batch_index+1)*BATCH_SIZE]

print augmented_mnist_images.shape
print augmented_mnist_labels.shape
seed = np.random.get_state()
np.random.shuffle(augmented_mnist_images)
np.random.set_state(seed)
np.random.shuffle(augmented_mnist_labels)

m_dim = 64
inputs = tf.keras.Input(shape=(32,32,1))
h1 = tf.keras.layers.Conv2D(1*m_dim, 3, 2, padding='same', activation=tf.nn.relu)(inputs)
h2 = tf.keras.layers.Conv2D(2*m_dim, 3, 2, padding='same', activation=tf.nn.relu)(h1)
h3 = tf.keras.layers.Conv2D(4*m_dim, 3, 2, padding='same', activation=tf.nn.relu)(h2)
h4 = tf.keras.layers.Conv2D(8*m_dim, 3, 2, padding='same', activation=tf.nn.relu)(h3)
h5 = tf.keras.layers.Conv2D(256, 2, 2, padding='valid', activation=tf.nn.relu)(h4)
f = tf.keras.layers.Flatten()(h5)
outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(f)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(augmented_mnist_images, augmented_mnist_labels, BATCH_SIZE, EPOCHS, validation_split=0.1)
model.save('data/keras_mnist_classifier.h5', include_optimizer=False)

