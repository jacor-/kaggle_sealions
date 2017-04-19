# This line solves some minor problems when you do not have propery set the PYTHONPATH
exec(compile(open("fix_paths.py", "rb").read(), "fix_paths.py", 'exec'))

import settings
import os
import pandas as pd
from common import dataset_loaders
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from patch_generators.pos_and_negative_fix_size import LabelEncoding, data_generator
import time

experiment_name = 'all_resnet_v0'

# We create the data structure we need
experiment_folder_name = 'patches_single_size'
os.system('mkdir -p %s/%s' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/models' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/logs' % (settings.DATAMODEL_PATH, experiment_folder_name))

OUTPUT_MODEL = '%s/%s/models/all_discriminator.hdf5' % (settings.DATAMODEL_PATH, experiment_folder_name)
LOGS_PATH    = '%s/%s/logs/%s' % (settings.DATAMODEL_PATH, experiment_folder_name, experiment_name)



image_size_nn = 48
num_valid_cases = 60
patch_size = 110
batch_size = 25
big_batch_size, valid_batch_size = 2500, 400

restart_valid_train = True


def load_labels(restart_valid_train):
    # Generate consistent train and validation sets
    ## Only do this if the train / validation has not been generated yet
    original_labels = dataset_loaders.groundlabels_dataframe()
    if restart_valid_train == True:
        os.system('rm -f %s/%s/train_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))
        os.system('rm -f %s/%s/valid_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))

    if not os.path.exists('%s/%s/train_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name)) and not os.path.exists('%s/%s/valid_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name)):
        cases = original_labels['image'].unique()
        np.random.shuffle(cases)
        train_data = original_labels[original_labels.image.isin(cases[num_valid_cases:])]
        valid_data = original_labels[~original_labels.image.isin(cases[num_valid_cases:])]

        train_data.to_csv('%s/%s/train_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name), index = False)
        valid_data.to_csv('%s/%s/valid_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name), index = False)
    train_data = pd.read_csv('%s/%s/train_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))
    valid_data = pd.read_csv('%s/%s/valid_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))
    return original_labels, train_data, valid_data

original_labels, train_data, valid_data = load_labels(restart_valid_train)

# Initialize these cases
## We add background as label to the rest of existing labels
existing_labels = np.concatenate([original_labels['class'].unique(), ['background']])
labelencoder = LabelEncoding(existing_labels)
data_augmentation = ImageDataGenerator(vertical_flip=True, horizontal_flip = True, zoom_range = 0.02, rotation_range=180)

train_generator = data_generator(data_augmentation, labelencoder, train_data, batch_size=big_batch_size, min_buffer_before_start = 5000, patch_size=patch_size, image_size_nn=image_size_nn)
valid_generator = data_generator(None, labelencoder, valid_data, batch_size=valid_batch_size,  min_buffer_before_start = 1000, patch_size=patch_size, image_size_nn=image_size_nn)



import logging
from sklearn import metrics
from keras import backend as K

K.set_image_dim_ordering('th')

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, History
from dl_utils.dl_networks.resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard


    

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
loss_history = History()

# Load model
model = ResnetBuilder().build_resnet_50((3,image_size_nn,image_size_nn),len(existing_labels))
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy')#,'fmeasure'])
#model.load_weights(OUTPUT_MODEL)



nb_epoch=1000
verbose=1
#class_weight={0:1., 1:4.},
validation_steps=2
steps_per_epoch = 5


min_val_loss = 10000
try:
    for i_epoch in range(nb_epoch):
        j_ep = 0

        t1 = time.time()
        loss_train = []
        for x, y in train_generator:
            if j_ep  < steps_per_epoch:
                j_ep += 1
                model.fit(x,y,verbose = 0, batch_size=batch_size,epochs=1,shuffle = False, callbacks=[loss_history])
                loss_train.append(np.mean(loss_history.history['loss']))
            else:
                break
        train_loss = np.mean(loss_train)

        # Validation stage
        losses = []
        j_valid = 0
        for x,y in valid_generator:
            if j_valid < validation_steps:
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
except:
    model.save_weights(OUTPUT_MODEL.split(".")[0] + "_interrupted_by_exception.hdf5")

print("EXITING!")
