# This line solves some minor problems when you do not have propery set the PYTHONPATH
exec(compile(open("fix_paths.py", "rb").read(), "fix_paths.py", 'exec'))

from common import dataset_loaders
import settings
import numpy as np
import pandas as pd
from skimage.feature import blob_dog, blob_log, blob_doh
import time

def load_predictions(casename, df):
    img = dataset_loaders.load_image(casename)
    pred = np.load("%s/%s_%s_%d.npz" % (annotations_path, annotation_basename, casename, scan_window))
    if casename in df['image'].unique():
        labels = df[df['image'] == casename]
    else:
        labels = None
    return img, pred['preds'], labels

def convert_to_binary_prediction(preds):
    # We map from any kind of car to one specific type
    return 1-preds[:,:,9]

def detect_blobs(preds):
    ## IN: 2d-greymap being 1 high probability of an element been there and 0 low probability
    ## Out: dataframe ['x','y','r']
    blobs_doh = blob_doh(preds, max_sigma=5, threshold=.01)
    detected = pd.DataFrame(blobs_doh[blobs_doh[:,2] > 1], columns = ['x','y','r'])
    detected.x = detected.x * scan_window + window_size / 2
    detected.y = detected.y * scan_window + window_size / 2
    return detected

def find_matches(casename, labels, detected, accepted_radius = 20):
    ## It take two dataframes:
    ### Labels: 'id', 'image', 'class', 'x', 'y'
    ### Detected: 'x', 'y', 'r'
    
    dfres = pd.DataFrame(columns = ['id','xref','yref','classref','xdet','ydet','rdet','pointtype'])

    ## TPositives
    for i_row in detected.index.values:
        if labels is None:
            # This is, for sure a FP
            xdet, ydet, radiusdet = detected.ix[i_row].x, detected.ix[i_row].y, detected.ix[i_row].r
            dfres.loc[dfres.shape[0]] = [casename, None, None, None, xdet, ydet, radiusdet,'FP']
        else:
            xdet, ydet, radiusdet = detected.ix[i_row].x, detected.ix[i_row].y, detected.ix[i_row].r
            dist_radius = ((labels[['x','y']]-detected.ix[i_row][['x','y']])**2).sum(axis=1).apply(np.sqrt)
            min_dist = dist_radius.min()
            if min_dist < accepted_radius:
                # This is a match. TP
                match_case = labels.ix[dist_radius.argmin()]
                xref, yref, classref = match_case.x, match_case.y, match_case['class']
                dfres.loc[dfres.shape[0]] = [casename, xref, yref, classref, xdet, ydet, radiusdet,'TP']
            else:
                # This case is a FP
                dfres.loc[dfres.shape[0]] = [casename, None, None, None, xdet, ydet, radiusdet,'FP']
    # If a point has not been matched to any point dref, it means it is a FN
    if labels is not None:
        for i_row in labels.index.values:
            if labels.ix[i_row].x not in dfres.xref.values:
                dfres.loc[dfres.shape[0]] = [casename, labels.ix[i_row].x, labels.ix[i_row].y, labels.ix[i_row]['class'],None, None, None, 'FN']
    return dfres


t1 = time.time()

# Prepare model parametrization

## Load the labels the same way we loaded them to run the first stage prediction
original_labels = dataset_loaders.groundlabels_dataframe()

## Set the same metaparameters we used for the first stage prediction
annotation_basename = 'sfinder'
model_name = 'patch_seal_finder'
annotations_path = "%s/%s/annotations" % (settings.DATAMODEL_PATH, model_name)
scan_window = 10
window_size = 80

## Parametrize the matches:
### 1 - Maximum accepted distance between reference and detection to consider it a Hit/Miss
### 2 - Define the function we use to go from the original prediction to a prob (different if it is multiclass or monoclass)
max_accepted_dist = 20
convert_to_binary_label = convert_to_binary_prediction
full_pred_filename = 'full_prediction_summary.csv'
fp_filename = 'FPs.csv'



# Verify the predictions over the whole dataset.

## Run case per case
predictions = []
for casename in dataset_loaders.get_casenames():
    try:
        img, preds, labels = load_predictions(casename, original_labels)
        print("Doing case %s" % casename)
    except:
        #print("%s not ready" % casename)
        continue
    preds = convert_to_binary_label(preds)
    detected = detect_blobs(preds)
    dfres = find_matches(casename, labels, detected, accepted_radius = max_accepted_dist)
    predictions.append(dfres)

## Save all the predictions (TP, TN, FPs) in a folder
predictions = pd.concat(predictions)
predictions.to_csv(annotations_path + "/" + full_pred_filename, header = False, index = False)

def generate_label(row):
    if row['pointtype'] == 'TP':
        return row['classref']
    elif row['pointtype'] == 'FP':
        return 'FP'
    elif row['pointtype'] == 'FN':
        return row['classref']

## Take only FPs and write them in a specific file
predictions['new_label'] = predictions.apply(lambda x: generate_label(x), axis = 1)
predictions = predictions[predictions['new_label']=='FP']
predictions[['id','new_label','xdet','ydet']].to_csv(annotations_path + '/' + fp_filename ,header=False, index=False)

print("Done in %0.0f seconds" % (t1-time.time()))