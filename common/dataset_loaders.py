import pandas as pd
import settings
import os
import numpy as np
from PIL import Image

destination_path = '%s/kaggle_forum_labels.csv' % settings.DATA_PATH
labels_url = 'https://raw.githubusercontent.com/gecrooks/sealionengine/master/outdir/coords.csv'

def groundlabels_dataframe():
    if not os.path.isfile(destination_path):
        os.system('wget -O %s %s' % (destination_path, labels_url))
    df = pd.read_csv(destination_path)
    df.columns = ['image','class','x','y']
    return df

# map category format:
#  map_category = {0:'type1', 1:'type2', 2:'type3', 3:'type4', 4:'type5'}
def map_labels(original_labels, map_category):
    glabels = original_labels.copy() 
    glabels['class'] = original_labels['class'].apply(lambda x: map_category[x])
    return glabels

### Images functionalities
def load_image(casename):
    try:
        return np.asarray(Image.open('%s/Train/%s.jpg' % (settings.DATA_PATH,casename))) / 255.
    except:
        return np.asarray(Image.open('%s/Test/%s.jpg' % (settings.DATA_PATH,casename))) / 255.

def get_casenames(test = False):
    if test:
        return [x[:-4] for x in os.listdir('%s/Test' % (settings.DATA_PATH)) if 'jpg' in x]
    else:
        return [x[:-4] for x in os.listdir('%s/Train' % (settings.DATA_PATH)) if 'jpg' in x]
