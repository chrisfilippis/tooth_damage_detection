# from customize import create_superpixels
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from sklearn.metrics import confusion_matrix

from preprocess_coco import get_superpixels, get_superpixels_new
from utils import merge_classes, merge_class_labels, plot_confusion_matrix, plot_data
from customize import custom_mappings
matplotlib.use('tkagg')

def balanced(C, adjusted=False, sample_weight=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        # warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score


# acc = np.array([[12489,   549,   113,  23,  27,   1,  35], [ 3,  18,   5,   0,   0,   0,   0], [ 4,  36,  10,   9,   0,   0,   0], [ 8,  14,  13,  14,   2,   0,   0], [ 2,   0,   2,   4,  22,   0,   0], [ 51,   0,  17,  23,   9,  35,  35], [ 0,   0,   0,   0,   0,   0,   0]])
# acc_custom = np.array([[12489,  662,   23,   63], [7,   69,    9,    0], [8,   27,   14,    2], [53,   19,   27,  101]])

# acc = np.array([[0,   549,   113,  23,  27,   1,  35], [ 3,  18,   5,   0,   0,   0,   0], [ 4,  36,  10,   9,   0,   0,   0], [ 8,  14,  13,  14,   2,   0,   0], [ 2,   0,   2,   4,  22,   0,   0], [ 51,   0,  17,  23,   9,  35,  35], [ 0,   0,   0,   0,   0,   0,   0]])
# acc_custom = np.array([[0,  662,   23,   63], [7,   69,    9,    0], [8,   27,   14,    2], [53,   19,   27,  101]])

acc = np.array([[844  , 0  ,19  , 8  , 4  , 0  , 0], [  0  , 0  , 0  , 0  , 0  , 0  , 0], [  1  , 0  , 0  , 0  , 0  , 0  , 0], [  0  , 0  , 8  ,15  , 0  , 0  , 0], [  1  , 0  , 2  , 0  , 0  , 0  , 0], [  0  , 0  , 0  , 0  , 0  , 0  , 0], [  0  , 0  , 0  , 0  , 0  , 0  , 0]])
# acc_custom = np.array([[844,  19,   8,   4], [  1,   0,   0,   0], [  0,   8,  15,   0], [  1,   2,   0,   0]])

# acc[0,0] = 0
# acc_custom[0,0] = 0

print(np.sum([acc[0, 0], acc[1,1], acc[2,2] ,acc[3,3] , acc[4,4], acc[5,5], acc[6,6]]) / np.sum(acc))
# print(np.sum(acc_custom))

# print(np.sum([acc[0, 0], acc[1,1], acc[2,2] ,acc[3,3] , acc[4,4], acc[5,5], acc[6,6]]))
# print(np.sum([acc_custom[0, 0], acc_custom[1,1], acc_custom[2,2] ,acc_custom[3,3]]))

print(balanced(acc))
# print(balanced(acc_custom))

plot_confusion_matrix(acc, classes=set(merge_classes()), normalize=False, title='Confusion matrix, without normalization')
plt.show()
# plot_confusion_matrix(acc_custom, classes=set(merge_classes(custom_mapping=custom_mappings())), normalize=False, title='Confusion matrix, without normalization')
# plt.show()
exit()
