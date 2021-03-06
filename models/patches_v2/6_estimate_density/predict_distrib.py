# This line solves some minor problems when you do not have propery set the PYTHONPATH
exec(compile(open("../fix_paths.py", "rb").read(), "../fix_paths.py", 'exec'))

import tensorflow as tf 
import keras
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


import settings
import os
import pandas as pd
from common import dataset_loaders
from pos_and_negative_fix_size import LabelEncoding
import numpy as np
import time

import logging
from sklearn import metrics
from keras import backend as K

K.set_image_dim_ordering('th')

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, History
from dl_utils.dl_networks.resnet import ResnetBuilder

import scipy.misc


########################################################
### Parameters
########################################################
experiment_folder_name = 'estimate_density'
experiment_name = 'resnet_v0'
annotations_name = '{path}/density_estimator_no_sum1_{casename}_{scan_window}.npz'
OUTPUT_MODEL = '%s/%s/models/density_estimator.hdf5' % (settings.DATAMODEL_PATH,experiment_folder_name)
image_size_nn = 45
patch_size = 80

batch_size = 2500
scan_step = 25

########################################################
### Parameters
########################################################
LOGS_PATH    = '%s/%s/logs/%s' % (settings.DATAMODEL_PATH, experiment_folder_name, experiment_name)
ANNOTATIONS_PATH = '%s/%s/annotations' % (settings.DATAMODEL_PATH, experiment_folder_name)
os.system('mkdir -p %s' % (ANNOTATIONS_PATH))

########################################################
### Function to load patches
########################################################
def scan_patches(img, image_size_nn, patch_size, step_frames, batch_size, square_to_scan = None):
    if square_to_scan is None:
        square_to_scan = [0,img.shape[0],0,img.shape[1]]
    # Ensure the patch does not go out of the image
    x_ini = np.max([square_to_scan[0], int(patch_size/2)])
    x_end = np.min([square_to_scan[1], int(img.shape[0]-patch_size/2)])
    y_ini = np.max([square_to_scan[2], int(patch_size/2)])
    y_end = np.min([square_to_scan[3], int(img.shape[1]-patch_size/2)])
    
    patches = []
    wnd = int(patch_size/2)
    for xi in range(x_ini, x_end, step_frames):
        for yi in range(y_ini, y_end, step_frames):
            patches.append(scipy.misc.imresize(img[xi-wnd:xi+wnd,yi-wnd:yi+wnd], [image_size_nn, image_size_nn]) / 255)
            if len(patches) == batch_size:
                #print(xi,yi, "  out of ", x_ini, x_end, y_ini, y_end)
                yield np.array(patches).transpose([0,3,1,2])
                patches = []
    if len(patches) > 0:
        print(xi,yi, "  out of ", x_ini, x_end, y_ini, y_end)
        yield np.array(patches).transpose([0,3,1,2])


########################################################
### Initilize things
########################################################

# We load the cases as they were originally...if needed
map_category = {0:0,1:1,2:2,3:3,4:4}
original_labels = dataset_loaders.groundlabels_dataframe()
original_labels = dataset_loaders.map_labels(original_labels, map_category)

## train_data = pd.read_csv('%s/%s/train_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))
## valid_data = pd.read_csv('%s/%s/valid_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))

# Initialize these cases
## We add background as label to the rest of existing labels
existing_labels = np.concatenate([original_labels['class'].unique()])
labelencoder = LabelEncoding(existing_labels)


########################################################
### Load model
########################################################
model = ResnetBuilder().build_resnet_50((3,image_size_nn,image_size_nn),len(existing_labels))
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy')#,'fmeasure'])
model.load_weights(OUTPUT_MODEL)




########################################################
### Run predictons
########################################################

def predict_case(case, model):
    t1 = time.time()
    img = dataset_loaders.load_image(case)
    patches = scan_patches(img, image_size_nn, patch_size, scan_step, batch_size, square_to_scan = None)
    preds = []
    for x in patches:
        preds.append(model.predict(x))
    preds = np.vstack(preds)
    
    # Reshape the image to the original shape, so we can map predictions to actual locations
    siz1 =  int((img.shape[0]-patch_size)/scan_step)+1
    siz2 =  int((img.shape[1]-patch_size)/scan_step)+1

    print(siz1,siz2, preds.shape[1])
    preds = preds.reshape([siz1,siz2,preds.shape[1]])
    return preds, time.time()-t1

#for casename in dataset_loaders.get_casenames():
for casename in ['616']:
    filename_to_save = annotations_name.format(path=ANNOTATIONS_PATH,scan_window=scan_step, casename=casename)
    if not os.path.isfile(filename_to_save):
        print("- Starting case %s with window step %d" % (casename, scan_step))
        preds, takentime = predict_case(casename, model)
        print("- Calculated case %s with window step %d in %0.0f seconds" % (casename, scan_step, takentime))
        np.savez_compressed(filename_to_save, preds = preds)
    else:
        print("- [ALREADY DONE] %s" % filename_to_save)
    break
# figure(); subplot(121); imshow(preds[:,:,1], cmap=cm.Greys); subplot(122); imshow(preds2[:,:,1], cmap = cm.Greys)

