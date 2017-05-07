import numpy as np
import scipy.misc

import os
from generator_settings import *
import time
import numpy as np

def get_case_to_load(folder):
    # return one casefilename if exists, otherwise wait until it does+
    while os.path.exists(folder) == False:
        #print("Path does not exist yet ", folder)
        time.sleep(1)
    avail_cases = [x for x in os.listdir(folder) if x[-4:] == '.npz']
    while len(avail_cases) == 0:
        print("Stuck... waiting for cases to be ready")
        time.sleep(10)
        avail_cases = [x for x in os.listdir(folder) if x[-4:] == '.npz']
    chosen_case = avail_cases[np.random.randint(len(avail_cases))]
    return chosen_case

def generatePatches_from_image(folder):
    # Load and remove the files after it's ready
    while(1):
        casename = get_case_to_load(folder)
        try: # Ugly code to avoid opening a npz file when it is not closed yet... stupid bug
            data = np.load(folder + '/' + casename)
            break
        except:
            time.sleep(1)
            continue
    Xs = [data['x_%d' % wind_size] for wind_size in WIND_SHAPES]
    Y = data['y']
    data.close()
    os.system('rm ' + folder + '/' + casename)
    return Xs,Y

    
def generate_chunks(folder, batch_size, min_buffer_before_start = 250):
    buffer_X = [[] for _ in WIND_SHAPES]
    buffer_Y = []
    while(1):
        X, Y = generatePatches_from_image(folder)
        if len(buffer_Y) == 0:
            buffer_X, buffer_Y = X, Y
        else:
            for i in range(len(WIND_SHAPES)):
                buffer_X[i] = np.vstack([X[i], buffer_X[i]])
            buffer_Y = np.concatenate([Y, buffer_Y])
        
        if len(buffer_Y) > min_buffer_before_start:
            # Shuffle before sending
            indexs = np.array(range(len(buffer_Y)))
            np.random.shuffle(indexs)
            for i in range(len(WIND_SHAPES)):
                buffer_X[i] = buffer_X[i][indexs]
            buffer_Y = buffer_Y[indexs]

            # Send and remove data!
            yield [buffer_X[i][:batch_size] for i in range(len(WIND_SHAPES))], buffer_Y[:batch_size]
            for i in range(len(WIND_SHAPES)):
                buffer_X[i] = buffer_X[i][batch_size:]
            buffer_Y = buffer_Y[batch_size:]

def data_generator(folder, dataaugmentation, batch_size, min_buffer_before_start = 200, image_size_nn = 45):
    for x,y in generate_chunks(folder, batch_size, min_buffer_before_start = min_buffer_before_start):
        for i in range(len(WIND_SHAPES)):
            if dataaugmentation is not None:
                for x_ in dataaugmentation.flow(x[i], batch_size = batch_size, shuffle = False):
                    x_ = np.array([scipy.misc.imresize(ss, [image_size_nn, image_size_nn]) / 255 for ss in x_])
                    break
            else:
                x_ = x[i]
            x[i] = x_.transpose([0,3,1,2])            
        yield x, y

'''
from files_of_data_creator import start_file_generator_process

# Give it some time to start pulling data
start_file_generator_process()
time.sleep(seconds_to_start_pulling_data)

# This is our generator
train_datagen = data_generator(folder_to_store_cases + '/train', None, batch_size, min_buffer_before_start = min_buffer_before_start, image_size_nn = image_size_nn)
for x, y in train_datagen:
    print("----")
    for i in range(len(WIND_SHAPES)):
        print(x[i].shape)
    print(y.shape)  
'''
