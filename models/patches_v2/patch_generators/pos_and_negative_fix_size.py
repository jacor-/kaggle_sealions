from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import scipy.misc
from common import dataset_loaders



# We generate a label ready for multiclass based on the string labels we use as input
class LabelEncoding(object):

    def __init__(self, input_classes):
        encoder = LabelEncoder()
        encoder.fit(input_classes)
        encoded_Y = encoder.transform(input_classes)
        lb = LabelBinarizer()
        lb.fit_transform(encoded_Y)
        
        self.encoder = encoder
        self.lb = lb

        self.num_classes = len(input_classes)
    
    def encode(self, x):
        label = self.lb.transform(self.encoder.transform(x)) 
        if self.num_classes == 2:
            return np.hstack((label, 1 - label))
        else:
            return label
            
    def get_class_id(self, label, binarized = True):
        if type(label) == np.ndarray:
            label = self.lb.classes_[np.argmax(label)]
        label = self.encoder.classes_[label]
        return label 
    

# We generate patches a punta pala
def _get_positive_patches_from_image(img, case, df, patch_size):
    X = []
    Y = []
    for ind in df[df['image'] == case].index:
        row = df.ix[ind]
        label, x, y = row['class'], row['x'], row['y']
        if  x - patch_size/2 >= 0 and x + patch_size/2 < img.shape[1] and y - patch_size/2 >= 0 and y + patch_size/2 < img.shape[1]:
            X.append(img[x-int(patch_size/2):x+int(patch_size/2), y-int(patch_size/2):y+int(patch_size/2),:])
            Y.append(label)
    return np.asarray(X, dtype = 'float32'), np.array(Y)

def _get_negative_patches_from_image(img, case, df, patch_size, quant_patches, label_to_use):
    car_coordinates = np.array(df[df['image'] == case][['x','y']].values)

    all_coords = []
    while len(all_coords) < quant_patches:
        new_coord = np.random.randint(2000-patch_size, size = [2]) + patch_size/2
        # we are far from all the other car coordinates
        if car_coordinates.shape[0] > 0:
            # BE CAREFUL!!! We divide by 3 to get some patches where there is a car, but not centered in the picture.
            # We want the network to be able to be able to distinguish location
            if (np.abs(car_coordinates - new_coord) > patch_size / 3).max(axis=1).all(): 
                all_coords.append(new_coord)
        else:
            all_coords.append(new_coord)
    patches = []
    for x,y in all_coords:
        patches.append(img[int(x-patch_size/2):int(x+patch_size / 2),int(y-patch_size/2):int(y+patch_size / 2),:])

    return np.asarray(patches, dtype = 'float32'), np.array([label_to_use for i in range(len(patches))])

# We get false positive patches
def _get_false_positives_patches_(img, case, df_FP, patch_size, label_to_use):
    if df_FP is not None: # if we have a false positive dataframe
        X = []
        for ind in df_FP[df_FP['image'] == case].index:
            row = df_FP.ix[ind]
            x, y = int(row['x']), int(row['y'])
            if  x - patch_size/2 >= 0 and x + patch_size/2 < img.shape[1] and y - patch_size/2 >= 0 and y + patch_size/2 < img.shape[1]:
                X.append(img[x-int(patch_size/2):x+int(patch_size/2), y-int(patch_size/2):y+int(patch_size/2),:])
        return np.asarray(X, dtype = 'float32'), np.array([label_to_use for i in range(len(X))])
    else:
        return [], []

def generatePatches_from_image(imagename, df, patch_size, quant_negative_patches = 25, negative_patch_label = 'background', df_FP = None):
    img = dataset_loaders.load_image(imagename)
    X_pos, Y_pos = _get_positive_patches_from_image(img, imagename, df, patch_size)
    X_neg, Y_neg = _get_negative_patches_from_image(img, imagename, df, patch_size, quant_negative_patches, negative_patch_label)
    X_fp, Y_fp = _get_false_positives_patches_(img, imagename, df_FP, patch_size, negative_patch_label)
    del img
    
    # Merge positive patches and negative patches.
    # There are always negative ones, so we start there and concatenate the positive and fp in case there are some
    X,Y = X_neg, Y_neg
    if len(Y_pos) != 0:
        X,Y = np.vstack([X_pos, X]), np.concatenate([Y_pos, Y])
    if len(X_fp) != 0:
        X,Y = np.vstack([X_fp, X]), np.concatenate([Y_fp, Y])
    
    # We return all the patches we have gathered
    return X,Y

    
def generate_chunks(df, batch_size, patch_size, df_FP = None, min_buffer_before_start = 250, neg_patches = 25):
    buffer_X = []
    buffer_Y = []
    while(1):
        imagename = df['image'].ix[np.random.randint(df.shape[0])]
        X, Y = generatePatches_from_image(imagename, df, patch_size, quant_negative_patches = neg_patches, negative_patch_label = 'background', df_FP = df_FP)
        if len(buffer_X) == 0:
            buffer_X, buffer_Y = X, Y
        else:
            buffer_X = np.vstack([X, buffer_X])
            buffer_Y = np.concatenate([Y, buffer_Y])
        
        if len(buffer_X) > min_buffer_before_start:
            # Start sending!
            indexs = np.array(range(len(buffer_X)))
            np.random.shuffle(indexs)
            buffer_X, buffer_Y = buffer_X[indexs], buffer_Y[indexs]
            yield buffer_X[:batch_size], buffer_Y[:batch_size]
            buffer_X, buffer_Y = buffer_X[batch_size:], buffer_Y[batch_size:]


def data_generator(dataaugmentation, labelencoder, df, batch_size, patch_size, df_FP = None, min_buffer_before_start = 200, neg_patches = 15, image_size_nn = 48):
    for x, y in generate_chunks(df, batch_size, patch_size, df_FP = df_FP, min_buffer_before_start = min_buffer_before_start, neg_patches = neg_patches):
        if dataaugmentation is not None:
            for x, y in dataaugmentation.flow(x, y, batch_size = batch_size, shuffle = False):
                break
        ## TODO If I want to do many rescales, here is the point!
        x = np.array([scipy.misc.imresize(ss, [image_size_nn, image_size_nn]) / 255 for ss in x])
        yield x.transpose([0,3,1,2]), labelencoder.encode(y)
