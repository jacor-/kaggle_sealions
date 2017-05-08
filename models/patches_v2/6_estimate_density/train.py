


# This line solves some minor problems when you do not have propery set the PYTHONPATH
exec(compile(open("../fix_paths.py", "rb").read(), "fix_paths.py", 'exec'))

import tensorflow as tf 
import keras
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from files_of_data_creator import start_file_generator_process
from generator_settings import *
from network_data_generator import data_generator
import settings
import os
import pandas as pd
from common import dataset_loaders
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
import logging
from sklearn import metrics
from keras import backend as K

K.set_image_dim_ordering('th')

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, History
from dl_utils.dl_networks.resnet import ResnetBuilder
from keras import losses
#from dl_utils.tb_callback import TensorBoard


## We create the data structure we need
os.system('mkdir -p %s/%s' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/models' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/logs' % (settings.DATAMODEL_PATH, experiment_folder_name))
LOGS_PATH    = '%s/%s/logs/%s' % (settings.DATAMODEL_PATH, experiment_folder_name, experiment_folder_name)


## Lets create the data generatorss
start_file_generator_process()
time.sleep(seconds_to_start_pulling_data)
# This is our generator
data_augmentation = ImageDataGenerator(vertical_flip=True, horizontal_flip = True, zoom_range = 0.00, rotation_range=180)
train_generator = data_generator(folder_to_store_cases + '/train', data_augmentation, big_batch_size_train, min_buffer_before_start = min_buffer_before_start_train, image_size_nn = image_size_nn)
valid_generator = data_generator(folder_to_store_cases + '/valid', data_augmentation, big_batch_size_valid, min_buffer_before_start = min_buffer_before_start_valid, image_size_nn = image_size_nn)





## Tensormierder

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

#tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
loss_history = History()

# Load model
model = ResnetBuilder().build_resnet_50((3,image_size_nn,image_size_nn),len(class_to_index)+1)
model.compile(optimizer=Adam(lr=1e-4), metrics = [losses.kullback_leibler_divergence], loss=losses.kullback_leibler_divergence)#,'fmeasure'])
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='val_kullback_leibler_divergence', mode='max', save_best_only=True)

#model.load_weights(INPUT_MODEL)



nb_epoch=1000
verbose=1
validation_steps_per_epoch = 1
train_steps_per_epoch      = 1


min_val_loss = 10000
#try:
if True:
    for i_epoch in range(nb_epoch):
        j_ep = 0

        t1 = time.time()
        loss_train = []
        for x, y in train_generator:
            print('SHAPES', x[0].shape, y.shape)
            if j_ep  < train_steps_per_epoch:
                j_ep += 1
                model.fit(x[0],y,verbose = 0, batch_size=batch_size,epochs=1,shuffle = False, callbacks=[loss_history])
                loss_train.append(np.mean(loss_history.history['loss']))
            else:
                break
        train_loss = np.mean(loss_train)

        # Validation stage
        losses = []
        j_valid = 0
        for x,y in valid_generator:
            if j_valid < validation_steps_per_epoch:
                losses.append( model.evaluate(x,y, verbose=0))
                j_valid += 1
            else:
                break
        valid_loss = np.mean(losses)

        # Save weights
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            model.save_weights(OUTPUT_MODEL)
        print("Epoch %d   -  valid loss: %0.3f   -   train loss: %0.3f    - Time %0.2f" % (i_epoch, valid_loss, train_loss, time.time()-t1))
#except:
#    model.save_weights(OUTPUT_MODEL.split(".")[0] + "_interrupted_by_exception.hdf5")

print("EXITING!")
