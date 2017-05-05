
# This line solves some minor problems when you do not have propery set the PYTHONPATH
exec(compile(open("fix_paths.py", "rb").read(), "fix_paths.py", 'exec'))

import tensorflow as tf 
import keras
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


import settings
import os
import pandas as pd
from common import dataset_loaders
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from patch_generators.pos_and_negative_fix_size import LabelEncoding, data_generator
import time

# Inicialization

## Pretrained model... I think it will be good to start from here and not from scratch
INPUT_MODEL = '%s/patch_seal_finder/models/seal_finder.hdf5' % (settings.DATAMODEL_PATH)

## Name of the new experiment and the new file where we will save the new output model
experiment_folder_name = 'seal_finder_fps'
experiment_name = 'fps_two_classes' # This one will only be used for the logs
OUTPUT_MODEL = '%s/%s/models/fps_seal_finder.hdf5' % (settings.DATAMODEL_PATH, experiment_folder_name)

## Provide data to locate the fp file if there is a FP discriminator available
fp_file_path = "%s/patch_seal_finder/annotations/FPs.csv" % (settings.DATAMODEL_PATH)
#fp_file_path = None

# Parameters
map_category = {0:'sealion', 1:'sealion', 2:'sealion', 3:'sealion', 4:'sealion'}
restart_valid_train = False

num_valid_cases = 60 # cases to exclude from train
image_size_nn = 50
patch_size = 80
batch_size = 25
big_batch_size, valid_batch_size = 5000, 500
neg_patches = 45

min_buffer_before_start_train, min_buffer_before_valid = 20000, 5000

# We create the data structure we need
os.system('mkdir -p %s/%s' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/models' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/logs' % (settings.DATAMODEL_PATH, experiment_folder_name))
LOGS_PATH    = '%s/%s/logs/%s' % (settings.DATAMODEL_PATH, experiment_folder_name, experiment_name)

# Load the FP dataframe
if fp_file_path is not None:
    fps_data = pd.read_csv(fp_file_path, header =None)
    fps_data.columns = ['image','fp','x','y']
else:
    fps_data = None # We can also run this without FP examples

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
existing_labels = np.concatenate([original_labels['class'].unique(), ['background']])
labelencoder = LabelEncoding(existing_labels)
data_augmentation = ImageDataGenerator(vertical_flip=True, horizontal_flip = True, zoom_range = 0.02, rotation_range=180)

train_generator = data_generator(data_augmentation, labelencoder, train_data, batch_size=big_batch_size, min_buffer_before_start = min_buffer_before_start_train, patch_size=patch_size, image_size_nn=image_size_nn, df_FP = fps_data, neg_patches = neg_patches)
valid_generator = data_generator(None, labelencoder, valid_data, batch_size=valid_batch_size,  min_buffer_before_start = min_buffer_before_valid, patch_size=patch_size, image_size_nn=image_size_nn, df_FP = fps_data, neg_patches = neg_patches)



import logging
from sklearn import metrics
from keras import backend as K

K.set_image_dim_ordering('th')

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, History
from dl_utils.dl_networks.resnet import ResnetBuilder
#from dl_utils.tb_callback import TensorBoard


    

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

#tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
loss_history = History()


# Load model
model = ResnetBuilder().build_resnet_50((3,image_size_nn,image_size_nn),len(existing_labels))
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy')#,'fmeasure'])
model.load_weights(INPUT_MODEL)




print(len(existing_labels), existing_labels)

nb_epoch=1000
verbose=1
#class_weight={0:1., 1:4.},
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
            if j_ep  < train_steps_per_epoch:
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
