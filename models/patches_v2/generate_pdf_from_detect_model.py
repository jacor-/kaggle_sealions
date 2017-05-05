exec(compile(open("fix_paths.py", "rb").read(), "fix_paths.py", 'exec'))  
import settings

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pylab import *
import pylab
import os
import pandas as pd
from common import dataset_loaders


########################################################
### Parameters
########################################################
experiment_folder_name = 'patch_seal_finder'
experiment_name = 'resnet_v0'
OUTPUT_MODEL = '%s/%s/models/seal_finder_remote.hdf5' % (settings.DATAMODEL_PATH, experiment_folder_name)

image_size_nn = 50
patch_size = 80

batch_size = 2500
scan_step = 20

########################################################
### Parameters
########################################################
ANNOTATIONS_PATH = '%s/%s/annotations' % (settings.DATAMODEL_PATH, experiment_folder_name)
ANNOTATIONS_SUMMARY_PATH = '%s/%s' % (settings.DATAMODEL_PATH, experiment_folder_name)

########################################################
### Initilize things
########################################################

# We load the cases as they were originally...if needed
map_category = {0:'sealion',1:'sealion',2:'sealion',3:'sealion',4:'sealion'}
original_labels = dataset_loaders.groundlabels_dataframe()
original_labels = dataset_loaders.map_labels(original_labels, map_category)


########################################################
### Run predictions and save file
########################################################

annotated_files = os.listdir(ANNOTATIONS_PATH)


def plot_cae(case_to_plot):
    file_id = annotated_files[case_to_plot]
    caseid = file_id.split("_")[1]
    img = dataset_loaders.load_image(caseid)
    detection = np.load(ANNOTATIONS_PATH+'/'+file_id)['preds']
    labels = original_labels[original_labels.image == int(caseid)]


    fig = figure()

    w, h = 15, 22
    fig.set_size_inches(w, h)


    title("EIS!")
    subplot(211)
    imshow(detection[:,:,0], cmap = cm.Reds)
    #plot((labels.y-patch_size/2)/scan_step,(labels.x-patch_size/2)/scan_step,'xr')

    subplot(212)
    imshow(img)
    plot(labels.y,labels.x,'xr')

    del img
    del detection

    print("My first stage prediction and groundtruth for case " + str(caseid))



with PdfPages('%s/output_pdf_%s.pdf' % (ANNOTATIONS_SUMMARY_PATH, OUTPUT_MODEL.split("/")[-1].split(".")[0])) as pdf:

    for case_to_plot in range(len(annotated_files)):
        plot_cae(case_to_plot)
        suptitle("Case " + str(annotated_files[case_to_plot]))
        pdf.savefig()  # saves the current figure into a pdf page
        close()
