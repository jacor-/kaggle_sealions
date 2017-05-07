
# This line solves some minor problems when you do not have propery set the PYTHONPATH
exec(compile(open("../fix_paths.py", "rb").read(), "../fix_paths.py", 'exec'))
import settings
from common import dataset_loaders
import numpy as np


experiment_folder_name = 'estimate_density'
OUTPUT_MODEL = '%s/%s/models/density_estimator.hdf5' % (settings.DATAMODEL_PATH, experiment_folder_name)


batch_size = 20
seconds_to_start_pulling_data = 150
image_size_nn = 45
num_images_for_validation = 50
big_batch_size_train, big_batch_size_valid = 5000, 500
min_buffer_before_start_train, min_buffer_before_start_valid = 10000, 1000


annotation_basename, model_name = 'sfinder', 'patch_seal_finder'
annotation_scan_window, annotation_window_size = 20, 80

## Define how many samples we are going to collect
RANDOM_SAMPLES_PER_IMAGE = 1000000 # Samples we are going to throw
MAX_NUMBER_OF_FILES_PER_IMAGE, SAMPLES_PER_FILE = 10, 200 # MAX 1000 samples per file

## Define what we will consider a candidate zone, wind shapes and so on
thrs = 7
class_to_index = {0:0,1:1,2:2,3:3,4:4}
#WIND_SHAPES = [40,60,80,100]
WIND_SHAPES = [80]


cases = dataset_loaders.get_casenames()
#np.random.shuffle(cases)
cases=sorted(cases)  # This way we can reproduce train and test easily
## Define how the generation process will behave
TRAIN_MAX_CASES_IN_FOLDER = 200
TRAIN_CASES_TO_GENERATE_PER_EXECUTION = 5
TRAIN_NUM_WORKERS = 2
#TRAIN_cases = ['616']
TRAIN_cases = cases[:-num_images_for_validation]
VALID_MAX_CASES_IN_FOLDER = 50
VALID_CASES_TO_GENERATE_PER_EXECUTION = 5
VALID_NUM_WORKERS = 1
#VALID_cases = ['616']
VALID_cases = cases[-num_images_for_validation:]


## Load the labels the same way we loaded them to run the first stage prediction
original_labels = dataset_loaders.groundlabels_dataframe()
annotations_path = "%s/%s/annotations" % (settings.DATAMODEL_PATH, model_name)

## path_to_store_files
folder_to_store_cases = '/fjord/temporal_patches_data'
