from __future__ import absolute_import
from keras.models import Model, load_model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Flatten, Dense, Activation, ZeroPadding2D, Dropout, Cropping2D
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import os
from time import time
#from keras.callbacks import ModelCheckpoint
#from dl_utils.tb_callback import TensorBoard
K.set_image_dim_ordering('th')

#import logging
#logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s  %(levelname)-8s %(message)s',
#                    datefmt='%m-%d %H:%M:%S')


def weighted_loss(y_true, y_pred, pos_weight=100, epsilon=1e-9):
    #if this argument is greater than 1 we will penalize more the nodules not detected as nodules, we can set it up to 10 or 100?
    y_true_f = K.flatten(y_true)  # y_true.flatten()
    y_pred_f = K.flatten(y_pred)  # y_pred.flatten()
    y_pred_f = K.clip(y_pred_f, epsilon, 1-epsilon) #clipping away from 0 and 1 to avoid NAN in loss computation
    return K.mean(-(1-y_true_f)*K.log(1-y_pred_f)-y_true_f*K.log(y_pred_f)*pos_weight)

class ThickUNET(object):
    def __init__(self,dropout=True, initialdepth=16, input_shape=(5,512,512), activation='relu',
     init='glorot_normal', saved_file=None, pos_weight=100):

        self.thickness = input_shape[0]
        self.model = self._get_model(input_shape, activation, init, initialdepth, dropout)        
        self.model.compile(optimizer=Adam(lr=1.0e-5), loss=weighted_loss, metrics=[weighted_loss])

        if saved_file is not None:
            try:
                self.model.load_weights(saved_file)
            except:
                print 'EXPECTED MODEL'+self.model.get_config()
                print '-------------------'
                print 'SAVED MODEL'+load_model(saved_file, custom_objects={'weighted_loss': weighted_loss}).get_config()
                raise Exception("WARNING: the file does not contain a model matching this arquitecture!!")


    def _get_model(self,inp_shape, activation='relu', init='glorot_normal', first_depth=32,dropout=False):
        inputs = Input(inp_shape)

        conv1 = Convolution2D(first_depth, 3, 3, activation=activation, init=init, border_mode='same')(inputs)
        conv1 = Convolution2D(first_depth, 3, 3, activation=activation, init=init, border_mode='same')(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(first_depth*2, 3, 3, activation=activation, init=init, border_mode='same')(pool1)
        conv2 = Convolution2D(first_depth*2, 3, 3, activation=activation, init=init, border_mode='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(first_depth*4, 3, 3, activation=activation, init=init, border_mode='same')(pool2)
        conv3 = Convolution2D(first_depth*4, 3, 3, activation=activation, init=init, border_mode='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(first_depth*8, 3, 3, activation=activation, init=init, border_mode='same')(pool3)
        conv4 = Convolution2D(first_depth*8, 3, 3, activation=activation, init=init, border_mode='same')(conv4)
        if dropout:
            conv4 = Dropout(0.5)(conv4) #using valid dropout as in the paper
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(first_depth*16, 3, 3, activation=activation, init=init, border_mode='same')(pool4)
        conv5 = Convolution2D(first_depth*16, 3, 3, activation=activation, init=init, border_mode='same')(conv5)
        if dropout:
            conv5 = Dropout(0.5)(conv5) #using valid dropout as in the paper

        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
        conv6 = Convolution2D(first_depth*8, 3, 3, activation=activation, init=init, border_mode='same')(up6)
        conv6 = Convolution2D(first_depth*8, 3, 3, activation=activation, init=init, border_mode='same')(conv6)

        up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
        conv7 = Convolution2D(first_depth*4, 3, 3, activation=activation, init=init, border_mode='same')(up7)
        conv7 = Convolution2D(first_depth*4, 3, 3, activation=activation, init=init, border_mode='same')(conv7)

        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
        conv8 = Convolution2D(first_depth*2, 3, 3, activation=activation, init=init, border_mode='same')(up8)
        conv8 = Convolution2D(first_depth*2, 3, 3, activation=activation, init=init, border_mode='same')(conv8)

        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
        conv9 = Convolution2D(first_depth, 3, 3, activation=activation, init=init, border_mode='same')(up9)
        conv9 = Convolution2D(first_depth, 3, 3, activation=activation, init=init, border_mode='same')(conv9)

        conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

        return Model(input=inputs, output=conv10)  # conv10

#----------------------------------------------    
# hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

