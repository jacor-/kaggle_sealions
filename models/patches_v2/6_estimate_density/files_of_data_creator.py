# This line solves some minor problems when you do not have propery set the PYTHONPATH
exec(compile(open("../fix_paths.py", "rb").read(), "../fix_paths.py", 'exec'))

import numpy as np
import scipy.ndimage.filters as fi
import pandas as pd
from generator_settings import *
import time
from concurrent import futures 

def load_predictions(casename, df, class_to_index, scan_window, window_size):
    NUM_CLASSES = len(list(class_to_index.keys()))
    def reconstruct_original(img, preds):
        rec_im = np.zeros(img.shape[:2])
        for i in range(int(window_size/2),img.shape[0]-int(window_size/2),scan_window):
            for j in range(int(window_size/2),img.shape[1]-int(window_size/2),scan_window):
                value = preds[int(int(i-window_size/2) / scan_window), int(int(j-window_size/2) / scan_window)][0]
                rec_im[i-int(window_size/2):i+int(window_size/2),j-int(window_size/2):j+int(window_size/2)] += value
        return rec_im

    def get_label_density(img, label):

        def gkern2(inp, nsig=20):
            """Returns a 2D Gaussian kernel array normalized to the maximum of the response"""
            imp_resp = np.zeros([100,100]); imp_resp[50,50] = 1 ; z = fi.gaussian_filter(imp_resp, nsig) # maximum of the impulse response
            norm_value = z.max() * 0.95 # We soft the function so the 1s area is a bit bigger
            ret = fi.gaussian_filter(inp, nsig) / norm_value 
            return np.clip(ret, 0., 1) # each normal will peak at 1... but we can add them and have values higher than 1! We simply clip it

        rec_im = np.zeros([img.shape[0], img.shape[1], NUM_CLASSES])
        for i in labels.index.values:
            row = labels.ix[i]
            rec_im[row.x,row.y,class_to_index[row['class']]] = 1
        for i_class in range(NUM_CLASSES):
            rec_im[:,:,i_class] = gkern2(rec_im[:,:,i_class])
        return np.nan_to_num(rec_im)

    img = dataset_loaders.load_image(casename)
    pred = np.load("%s/%s_%s_%d.npz" % (annotations_path, annotation_basename, casename, scan_window))
    if int(casename) in df['image'].unique():
        labels = df[df['image'] == int(casename)]
        lab_dens = get_label_density(img, labels)
    else:
        labels = None
    return img, reconstruct_original(img, pred['preds']), lab_dens


def generate_candidates(N_samples, candidate_space):
    candidates = []
    for xcan,ycan in zip(np.random.randint(candidate_space.shape[0], size=N_samples),
                         np.random.randint(candidate_space.shape[1], size=N_samples)):
        if candidate_space[xcan,ycan]:
            candidates.append([xcan, ycan])
    return candidates


def samples_from_coordinates(img, lab_dens, coordinates, batch_size = 20, wind_shapes = [80, 60, 100]):
    X = [np.zeros([batch_size, 3,wind_, wind_]) for wind_ in wind_shapes] # A different array for each input
    Y = np.zeros([batch_size, lab_dens.shape[2]])
    max_wind = np.max(wind_shapes)
    current_index = 0
    for i_coord in range(len(coordinates)):
        i, j = coordinates[i_coord][0], coordinates[i_coord][1]
        if      i-int(max_wind/2) >= 0 and \
                j-int(max_wind/2) >= 0 and \
                i+int(max_wind/2) < img.shape[0] and \
                j+int(max_wind/2) < img.shape[1]:
            Y[current_index] = lab_dens[i,j]
            for iwind, wind_ in enumerate(wind_shapes):
                X[iwind][current_index] = img[i-int(wind_/2):i+int(wind_/2),j-int(wind_/2):j+int(wind_/2),:].transpose(2,0,1)
            current_index += 1
        if current_index == batch_size:
            yield X, Y
            current_index = 0
    if current_index != 0:
        yield [X[iwind][:current_index] for iwind in range(len(wind_shapes))], Y[:current_index]


def generate_samples_given_case(case, folder):
    print('generating case %s  <-- %s' % (case, folder))

    # Load image and preditions
    img, rec_im, lab_dens = load_predictions(case, original_labels, class_to_index, annotation_scan_window, annotation_window_size)
    # Define the threshold to consider candidates or not
    candidate_space = rec_im>thrs
    # Define the threshold to consider candidates or not
    samples = generate_candidates(RANDOM_SAMPLES_PER_IMAGE, candidate_space)[:MAX_NUMBER_OF_FILES_PER_IMAGE*SAMPLES_PER_FILE]
    # Generate samples and save them into files
    samples_generator = samples_from_coordinates(img, lab_dens, samples, batch_size = SAMPLES_PER_FILE, wind_shapes = WIND_SHAPES)

    for x, y in samples_generator:
        # discard the last one if
        if y.shape[0] == SAMPLES_PER_FILE:
            sv = {}
            for i in range(len(WIND_SHAPES)):
                sv['x_%d' % WIND_SHAPES[i]] = x[i]
            sv['y'] = y
            np.savez(folder+'/'+case+'_'+str(np.random.randint(100000)), **sv)



def prepare_data_files_and_monitor(folder, available_cases,
    CASES_TO_GENERATE_PER_EXECUTION, NUM_WORKERS, MAX_CASES_IN_DATA_FOLDER):
    executor = futures.ThreadPoolExecutor(max_workers=NUM_WORKERS)
    current_case = []
    # Delete the folder if it already exists and create if from scratch
    os.system('rm -rf %s' % folder)
    os.system('mkdir %s' % folder)

    while(1):
        if len(os.listdir(folder)) < MAX_CASES_IN_DATA_FOLDER: # generate more data only if the max is not reached
            current_cases = []
            while len(current_cases) < CASES_TO_GENERATE_PER_EXECUTION:
                case = available_cases[np.random.randint(len(available_cases))]
                current_cases.append(executor.submit(generate_samples_given_case,case, folder))
            futures.wait(current_cases, return_when=futures.ALL_COMPLETED)
        else:
            time.sleep(5)

def start_file_generator_process():
    ## Create a generator which starts creating files of data
    os.system('rm -rf %s' % folder_to_store_cases)
    os.system('mkdir %s' % folder_to_store_cases)
    train_async_gen = futures.ThreadPoolExecutor(max_workers=1).submit( prepare_data_files_and_monitor, 
                                                                        folder_to_store_cases + '/train', 
                                                                        TRAIN_cases, 
                                                                        TRAIN_CASES_TO_GENERATE_PER_EXECUTION, 
                                                                        TRAIN_NUM_WORKERS, 
                                                                        TRAIN_MAX_CASES_IN_FOLDER)
    valid_async_gen = futures.ThreadPoolExecutor(max_workers=1).submit( prepare_data_files_and_monitor, 
                                                                        folder_to_store_cases + '/valid', 
                                                                        VALID_cases, 
                                                                        VALID_CASES_TO_GENERATE_PER_EXECUTION, 
                                                                        VALID_NUM_WORKERS, 
                                                                        VALID_MAX_CASES_IN_FOLDER)
    return train_async_gen, valid_async_gen

#t,v = start_file_generator_process()

