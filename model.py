from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D,Lambda
from keras.layers.advanced_activations import LeakyReLU,ELU
from math import floor
import tensorflow as tf

#%% Image processing Function 
def resize(image):
    return tf.image.resize_images(image, (64, 128),tf.image.ResizeMethod.AREA)

def contrast(image):
    return tf.image.random_contrast(image,lower=0.2,upper=1.8) #adjust_contrast(image,contrast_factor=0.8)
    
def random_brightness(image):
    tf.image.random_brightness(image,max_delta=63)    

def linear_normalization(image):
    # Normalize the images to be between -0.5 and 0.5
    cast_image = tf.cast(image, tf.float32)
    return cast_image / 255. - 0.5

def L2_normalization(image):
    cast_image = tf.cast(image, tf.float32)
    return cast_image / tf.sqrt(tf.reduce_sum(cast_image**2,axis=1,keep_dims=True))

#%% Create the model
def create_model(image_shape = (160,320),keep_prob = 0.5):
    model = Sequential()

    shape = image_shape + (3,)
    # image preprocessing layers: adding random contrast, L2 normalization, Cropping the sky and hood, resize to (64,128)
    model.add(Lambda(contrast, input_shape=shape, output_shape=shape))
    model.add(Lambda(L2_normalization))

    model.add(Cropping2D(cropping=((30,25), (0,0))))
    model.add(Lambda(resize))

    # Color adjustment Conv
    model.add(Convolution2D(3, 1, 1, border_mode='valid', init='he_normal'))

    # First Conv layer
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(Activation('relu'))
    
    # Second Conv layer
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(Activation('relu'))
    
    # Third Conv layer
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(Activation('relu'))
    
    # Fourth Conv layer
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init='he_normal'))
    model.add(Activation('relu'))
    
    # Fifth Conv layer
    model.add(Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='valid', init='he_normal'))
    model.add(Activation('relu'))

    model.add(Flatten())

    # fully connected layer 1
    model.add(Dense(512, name='FC1', init='he_normal'))
    model.add(Activation(ELU()))
    model.add(Dropout(keep_prob))

    # fully connected layer 2
    model.add(Dense(64, name='FC2', init='he_normal'))
    model.add(Activation(ELU()))
    model.add(Dropout(keep_prob))

    # fully connected layer 3
    model.add(Dense(16, name='FC3', init='he_normal'))
    model.add(Activation(ELU()))
    model.add(Dropout(keep_prob))

    # fully connected layer 4
    model.add(Dense(1,name='y_pred', init='he_normal'))

    return model
