import numpy as np
import matplotlib
matplotlib.use('tkagg')
from itertools import groupby, product
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


init_mapping = {
        0:0,
        1:1,
        2:2,
        3:3,
        4:4,
        5:5,
        6:6 }

init_mapping_labels = {
        0:0,
        1:1,
        2:2,
        3:3,
        4:4,
        5:5,
        6:6 }

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def merge_classes(classes=None, custom_mapping=None):

    mapping = init_mapping

    if custom_mapping is not None:
        mapping = custom_mapping

    if classes is None:
        return list(set(mapping.values()))

    return [mapping[clas] for clas in classes]


def merge_class_labels(classes=None, custom_mapping=None):

    mapping = init_mapping_labels

    if custom_mapping is not None:
        mapping = custom_mapping

    if classes is None:
        return list(set(mapping.values()))

    return [mapping[clas] for clas in classes]


def plot_data(superpixels_classes, predictions, labels, classes):
    conf_matrix = confusion_matrix(superpixels_classes, predictions, labels=labels)

    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=classes, normalize=False, title='Confusion matrix, without normalization')
    plt.show()