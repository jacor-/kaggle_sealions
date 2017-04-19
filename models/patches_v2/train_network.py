# This line solves some minor problems when you do not have propery set the PYTHONPATH
exec(compile(open("fix_paths.py", "rb").read(), "fix_paths.py", 'exec'))

# Imports to ignore the thousands of keras warnings messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


import settings
import os
import pandas as pd
from common import dataset_loaders
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from patch_generators.pos_and_negative_fix_size import LabelEncoding, data_generator
import time

experiment_name = 'resnet_v0'

# We create the data structure we need
experiment_folder_name = 'patch_seal_finder'
os.system('mkdir -p %s/%s' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/models' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/logs' % (settings.DATAMODEL_PATH, experiment_folder_name))

OUTPUT_MODEL = '%s/%s/models/seal_finder.hdf5' % (settings.DATAMODEL_PATH, experiment_folder_name)
LOGS_PATH    = '%s/%s/logs/%s' % (settings.DATAMODEL_PATH, experiment_folder_name, experiment_name)

map_category = {0:'sealion', 1:'sealion', 2:'sealion', 3:'sealion', 4:'sealion'}


image_size_nn = 50
num_valid_cases = 60
patch_size = 80
batch_size = 25
neg_patches_per_image = 60
big_batch_size, valid_batch_size = 2500, 400
validation_steps=2
train_steps = 5

restart_valid_train = True


def load_labels(restart_valid_train, map_category = None):
    # Generate consistent train and validation sets
    ## Only do this if the train / validation has not been generated yet
    original_labels = dataset_loaders.groundlabels_dataframe()
    if map_category is not None:
        original_labels = dataset_loaders.map_labels(original_labels, map_category)


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

original_labels, train_data, valid_data = load_labels(restart_valid_train, map_category)

# Initialize these cases
## We add background as label to the rest of existing labels
existing_labels = np.concatenate([['background'], original_labels['class'].unique()])
labelencoder = LabelEncoding(existing_labels)
data_augmentation = ImageDataGenerator(vertical_flip=True, horizontal_flip = True, zoom_range = 0.02, rotation_range=180)

train_generator = data_generator(data_augmentation, labelencoder, train_data, batch_size=big_batch_size, min_buffer_before_start = 5000, patch_size=patch_size, image_size_nn=image_size_nn, neg_patches= neg_patches_per_image)
valid_generator = data_generator(None, labelencoder, valid_data, batch_size=valid_batch_size,  min_buffer_before_start = 1000, patch_size=patch_size, image_size_nn=image_size_nn,  neg_patches= neg_patches_per_image)





import logging
from sklearn import metrics
from keras import backend as K

K.set_image_dim_ordering('th')

from keras.optimizers import Adam
from keras.callbacks import History
from dl_utils.dl_networks.resnet import ResnetBuilder


    

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

loss_history = History()

# Load model
model = ResnetBuilder().build_resnet_50((3,image_size_nn,image_size_nn), len(existing_labels))
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy')#,'fmeasure'])
#model.load_weights(OUTPUT_MODEL)



nb_epoch=1000
verbose=1
#class_weight={0:1., 1:4.},


min_val_loss = 10000
#try:
for i_epoch in range(nb_epoch):
    j_ep = 0

    t1 = time.time()
    loss_train = []
    for x, y in train_generator:
        if j_ep  < train_steps:
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
#except:
#    model.save_weights(OUTPUT_MODEL.split(".")[0] + "_interrupted_by_exception.hdf5")

print("EXITING!")
