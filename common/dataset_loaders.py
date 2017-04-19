from PIL import Image
import os
import numpy as np
import pandas as pd
import settings


import itertools
import pandas as pd
import numpy as np
import os


#map_category = {'A':'Moto', 'B':'Car', 'C':'Car', 'D':'Car', 'E':'Car', 'F':'Car', 'G':'Car', 'H':'Van', 'I':'Bus'}


### Results dataframe functionalities
def groundlabels_dataframe():
    labels_data = pd.read_csv('%s/trainingObservations.csv' % settings.DATA_PATH)

    def melt_series(s):
        lengths = s.str.len().values
        flat = [i for i in itertools.chain.from_iterable(s.values.tolist())]
        idx = np.repeat(s.index.values, lengths)
        return pd.Series(flat, idx, name=s.name)

    labels_data = melt_series(labels_data.detections.apply(lambda x: x.split("|"))).to_frame().join(labels_data.drop('detections', 1)).reindex_axis(labels_data.columns, 1)
    labels_data = labels_data[labels_data.detections != 'None']
    labels_data['x'] = labels_data.detections.apply(lambda x: int(x.split(":")[1]))
    labels_data['y'] = labels_data.detections.apply(lambda x: int(x.split(":")[0]))
    labels_data = labels_data.drop('detections',1)
    labels_data['image'] = labels_data.image.apply(lambda x: x[:-4])
    labels_data.index = range(labels_data.shape[0])
    return labels_data

# map category format:
#  map_category = {'A':'Moto', 'B':'Car', 'C':'Car', 'D':'Car', 'E':'Car', 'F':'Car', 'G':'Car', 'H':'Van', 'I':'Bus'}
def map_labels(original_labels, map_category):
    glabels = original_labels.copy() 
    glabels['class'] = original_labels['class'].apply(lambda x: map_category[x])
    return glabels

### Images functionalities
def load_image(casename):
    try:
        return np.asarray(Image.open('%s/training/%s.jpg' % (settings.DATA_PATH,casename))) / 255.
    except:
        return np.asarray(Image.open('%s/test/%s.jpg' % (settings.DATA_PATH,casename))) / 255.

def get_casenames(test = False):
    if test:
        return [x[:-4] for x in os.listdir('%s/test' % (settings.DATA_PATH))]
    else:
        return [x[:-4] for x in os.listdir('%s/training' % (settings.DATA_PATH))]
