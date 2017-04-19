# This line solves some minor problems when you do not have propery set the PYTHONPATH
exec(compile(open("fix_paths.py", "rb").read(), "fix_paths.py", 'exec'))

import settings
import os
import pandas as pd
from common import dataset_loaders
from patch_generators.pos_and_negative_fix_size import LabelEncoding
import numpy as np
import time

import logging
from sklearn import metrics
from keras import backend as K

K.set_image_dim_ordering('th')

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, History
from dl_utils.dl_networks.resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard

import scipy.misc


########################################################
### Parameters
########################################################
experiment_folder_name = 'patches_single_size'
experiment_name = 'vehicle_empty_resnet_v0'
annotations_name = '{path}/all_discriminator_remote_{casename}_{scan_window}.npz'
OUTPUT_MODEL = '%s/%s/models/all_discriminator_remote.hdf5' % (settings.DATAMODEL_PATH, experiment_folder_name)

image_size_nn = 48
patch_size = 110

batch_size = 25000
scan_step = 10

########################################################
### Parameters
########################################################
LOGS_PATH    = '%s/%s/logs/%s' % (settings.DATAMODEL_PATH, experiment_folder_name, experiment_name)
ANNOTATIONS_PATH = '%s/%s/annotations' % (settings.DATAMODEL_PATH, experiment_folder_name)
os.system('mkdir -p %s' % (ANNOTATIONS_PATH))

########################################################
### Function to load patches
########################################################
def scan_patches(imagename, image_size_nn, patch_size, step_frames, batch_size, square_to_scan = None):
    img = dataset_loaders.load_image(imagename)
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
                yield np.array(patches).transpose([0,3,1,2])
                patches = []
    if len(patches) > 0:
        yield np.array(patches).transpose([0,3,1,2])


########################################################
### Initilize things
########################################################

# We load the cases as they were originally...if needed
original_labels = dataset_loaders.groundlabels_dataframe()
## train_data = pd.read_csv('%s/%s/train_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))
## valid_data = pd.read_csv('%s/%s/valid_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))

# Initialize these cases
## We add background as label to the rest of existing labels
existing_labels = np.concatenate([original_labels['class'].unique(), ['background']])
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

def predict_case(case):
    t1 = time.time()
    patches = scan_patches(case, image_size_nn, patch_size, scan_step, batch_size, square_to_scan = None)
    preds = []
    for x in patches:
        preds.append(model.predict(x))
    preds = np.vstack(preds)
    preds = preds.reshape([int(np.sqrt(preds.shape[0])),int(np.sqrt(preds.shape[0])), preds.shape[1]])
    return preds, time.time()-t1

for casename in dataset_loaders.get_casenames():
    filename_to_save = annotations_name.format(path=ANNOTATIONS_PATH,scan_window=scan_step, casename=casename)
    if not os.path.isfile(filename_to_save):
        preds, takentime = predict_case(casename)
        print("- Calculated case %s with window step %d in %0.0f seconds" % (casename, scan_step, takentime))
        np.savez_compressed(filename_to_save, preds = preds)
    else:
        print("- [ALREADY DONE] %s" % filename_to_save)
