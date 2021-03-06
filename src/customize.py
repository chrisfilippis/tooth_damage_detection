import os
import sys
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from operator import itemgetter 
from itertools import groupby, product
from mrcnn import visualize
from preprocess_coco import get_superpixels, get_superpixels_new
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from mrcnn.model import log


def create_superpixels(image=None, image_file="C:\\Projects\\tooth_damage_detection_deeplab\\data\\annotator\\training\\anaxristina37.jpg"):
    
    if (image is None):
        # load the image and convert it to a floating point data type
        image = img_as_float(io.imread(image_file))
    
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    #return slic(image, n_segments = 9, max_iter=9, sigma = .15)#.astype(np.uint8)

    # import matplotlib
    # matplotlib.use('tkagg')
    # import matplotlib.pyplot as plt

    return slic(image, n_segments = 900, max_iter=9, sigma = 0.15)#.astype(np.uint8)
            
    # # show the output of SLIC
    # fig = plt.figure("Superpixels -- %d segments" % (9))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(image, segments))
    # plt.axis("off")

    # # show the plots
    # plt.show()
    # return segments


def get_bbox(img):
    a = np.where(img != 0)
    #y1, x1, y2, x2
    return np.array([np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])])


def combine_masks_and_superpixels(masks, class_ids, superpixels):
    n_superpixels = max(max(x) for x in superpixels)
    n_superpixels = int(n_superpixels) + 1
    # print('Number of superpixels', n_superpixels)
    # print('Superpixels', superpixels.shape)
    
    # print('Number of masks', masks.shape[2])
    # print('Masks', masks.shape)

    superpixels_classes = []
    superpixels_bboxes = []
    print('loop for ', n_superpixels)
    # loop superpixels verticaly
    # for i in range(20):
    for i in range(n_superpixels):
        superpixel = superpixels == i
        # print('Superpixels: ', superpixels)
        # print('Superpixel: ', superpixel)
        # print('Superpixel volume: ', get_superpixel_volume(superpixel, superpixels, i))
        superpixel_classes = get_superpixel_classes(superpixel, masks, class_ids)

        # print('Superpixel classes found', superpixel_classes)

        if(len(superpixel_classes) > 0):
            final_class = decide_class(superpixel_classes)
            
            # print('Superpixel data', superpixel)
            # print('Superpixel ' + str(i) + ' class found ' +  str(final_class[0]))

            superpixels_classes.append(final_class[0])
            superpixel_bbox = get_bbox(superpixel)
            # print('superpixel_bbox ', superpixel_bbox)            
            # print('Superpixel volume: ', get_superpixel_volume(superpixel, superpixels, i))
            # print('superpixel_bbox: ', superpixel_bbox)
            superpixels_bboxes.append(superpixel_bbox)
        else:
            # print('Superpixel ' + str(i) + ' class not found')
            superpixels_classes.append(100)

        # print('Superpixel ' + str(i))

    print('n_superpixels', n_superpixels)
    print('test_superpixels_classes_1', len(superpixels_classes))
    
    # remove first item data
    superpixels_bboxes.remove(superpixels_bboxes[0])
    superpixels_classes[0] = 100

    superpixels_classes = np.array(superpixels_classes)

    print('test_superpixels_classes', len(superpixels_classes))

    # classes = np.unique(superpixels_classes[superpixels_classes != 100])
    classes = superpixels_classes[superpixels_classes != 100]
    
    print('classes', len(superpixels_classes[superpixels_classes != 100]))

    result_masks = np.zeros((superpixels.shape[0], superpixels.shape[1], classes.shape[0]), dtype=int)

    # loop superpixels verticaly
    actual_class_index = 0
    for i in range(n_superpixels):
        superpixel_class = superpixels_classes[i]
        
        if(superpixel_class == 100):
            continue

        superpixel = superpixels == i
    
        # sup_class = classes == superpixel_class
        # print('superpixel_class', superpixel_class)
        # print('classes', classes)
        # print('sup_class', sup_class)
        # sup_class = sup_class.astype(np.uint8)

        class_mask = np.zeros(classes.shape[0], dtype=int)
        class_mask[actual_class_index] = 1
        class_mask = class_mask.astype(np.bool)
        # print('class_mask', class_mask)
        result_masks[superpixel] = class_mask
        actual_class_index += 1

    superpixels_classes = superpixels_classes.astype(np.int32)

    superpixels_classes[superpixels_classes == 100] = 0
    superpixels_classes[0] = 0

    return [{
        'class_ids': classes,
        'masks': result_masks.astype(np.bool),
        'rois': np.array(superpixels_bboxes).astype(np.int32),
        'scores' : np.full((result_masks.shape[2],), 0.21).astype(np.float32),
        'superpixels_classes': superpixels_classes
    }]


def decide_class(superpixel_classes):
    result = []
    for key, temp in groupby(superpixel_classes, key = itemgetter(1)):
        result.append((key, sum(list(map(itemgetter(0), temp)))))
    
    return sorted(result, key=lambda tup: tup[1])[-1]


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


def get_superpixel_volume(superpixel, superpixels, indx=0):
    
    superpixel_volume = 0
    min_y = 1024
    min_x = 768
    max_y = 0
    max_x = 0

    for x in range(superpixel.shape[0]):
        # print(str(x), str(np.unique(superpixel[:,x])) + ' / ' + str(np.unique(superpixels[:,x])))

        # if(len(np.unique(superpixel[x])) > 1):
            # print(np.where(superpixel[x] == True))
            # print(np.unique(superpixel[x]))

        for y in range(superpixel.shape[1]):
            if(superpixel[x][y] == True):
                
                # print(str(indx) + ': ' + str(x) + '/' + str(y), superpixel[x][y])
                
                if(y > max_y):
                    max_y = y
                if(x > max_x):
                    max_x = x

                if(y < min_y):
                    min_y = y
                if(x < min_x):
                    min_x = x

                superpixel_volume = superpixel_volume + 1

    # print(superpixels)
    
    return superpixel_volume, max_x, max_y, min_x, min_y


def get_superpixel_classes_old(superpixel, masks, class_ids):

    result = []

    for mask_index in range(masks.shape[2]):
        
        class_id = class_ids[mask_index]
        # print('Searhing for mask with index: ' + str(mask_index) + ' with class: ' + str(class_id))
        mask = masks[:, :, mask_index]

        superpixel_values = get_superpixel_class_weight(superpixel, mask, class_id, mask_index)

        if(superpixel_values == 0):
            continue
        
        result.append((superpixel_values, class_id))

    return result


def get_superpixel_classes(superpixel, masks, class_ids):

    result = []

    for mask_index in range(masks.shape[2]):
        
        class_id = class_ids[mask_index]
        
        # print(class_ids)
        # print('Searhing for mask with index: ' + str(mask_index) + ' with class: ' + str(class_id))
        mask = masks[:, :, mask_index]
        # print(sum(mask[superpixel]))
        # print(mask[superpixel])
        mask[mask == 1] = True
        mask[mask == 0] = False
        # print(sum(mask[superpixel]))
        # print(mask[superpixel])

        superpixel_values = sum(mask[superpixel])

        if(superpixel_values == 0):
            continue

        result.append((superpixel_values, class_id))

    # print(superpixel)    
    return result


def get_superpixel_class_weight(superpixel, mask, class_id, mask_index):

    superpixel_point_data = []

    superpixel_temp = np.copy(superpixel)
    mask_temp = np.copy(mask)

    superpixel_temp = superpixel_temp.astype(int)

    mask_temp[mask_temp == True] = 1
    mask_temp[mask_temp == False] = 2

    result = []
    for p_x in range(superpixel_temp.shape[0]):

        temp_reult = superpixel_temp[p_x] == mask_temp[p_x]
        line_result = temp_reult[np.where(temp_reult == True)].shape[0]
        result.append(line_result)

    return np.sum(result)


def custom_mappings():
    return {
        0:0,
        1:1,
        2:1,
        3:2,
        4:3,
        5:3,
        6:3 }


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


def transform_masks_to_superpixel(results, original_image, original_image_annotation, return_metrics=False):

    print('annotation_file', original_image_annotation)

    r = results[0]
    # superpixels = create_superpixels(image=original_image)
    print('loading superpixels...')
    superpixels, superpixels_classes = get_superpixels_new(original_image_annotation)

    # print('2', np.where(np.array(superpixels_classes)==2))
    # print('1', np.where(np.array(superpixels_classes)==1))
    
    # print('superpixels found.', superpixels.shape)
    print(str(len(superpixels_classes)) + ' superpixels classes found.')

    # print(r["masks"][-1][0])
    # print(r["masks"].shape)

    print('combine_masks_and_superpixels...')
    result = combine_masks_and_superpixels(r['masks'].astype(np.uint8), r['class_ids'].astype(np.uint8), superpixels)
    predictions = result[0]['superpixels_classes']
    # print(result[0]["masks"][-1][0])
    # print(result[0]["masks"].shape)
    
    # print('2', np.where(predictions==2))
    # print('1', np.where(predictions==1))
    
    # print('predictions', predictions)
    # print('superpixels_classes', superpixels_classes)

    print('result', len(predictions))
    
    predictions = merge_classes(classes=predictions, custom_mapping=None)
    superpixels_classes = merge_classes(classes=superpixels_classes, custom_mapping=None)

    accuracy = accuracy_score(superpixels_classes, predictions)
    balanced_accuracy = balanced_accuracy_score(superpixels_classes, predictions)

    conf_matrix = confusion_matrix(superpixels_classes, predictions, labels=merge_class_labels())
    log("accuracy_score", accuracy)
    log("balanced_accuracy_score", balanced_accuracy)
    
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=set(merge_classes()), normalize=False, title='Confusion matrix, without normalization')
    plt.show()

    conf_matrix[0, 0] = 0

    log("accuracy_score no bg", np.sum([conf_matrix[0, 0], conf_matrix[1,1], conf_matrix[2,2] ,conf_matrix[3,3] , conf_matrix[4,4], conf_matrix[5,5], conf_matrix[6,6]]) / np.sum(conf_matrix))
    log("balanced_accuracy_score no bg", balanced(conf_matrix))

    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=set(merge_classes()), normalize=False, title='Confusion matrix, without normalization')
    plt.show()

    predictions = merge_classes(classes=predictions, custom_mapping=custom_mappings())
    superpixels_classes = merge_classes(classes=superpixels_classes, custom_mapping=custom_mappings())

    conf_matrix_custom = confusion_matrix(superpixels_classes, predictions, labels=merge_class_labels(custom_mapping=custom_mappings()))

    accuracy_custom = accuracy_score(superpixels_classes, predictions)
    balanced_accuracy_custom = balanced_accuracy_score(superpixels_classes, predictions)
    
    log("accuracy_score", accuracy_custom)
    log("balanced_accuracy_score", balanced_accuracy_custom)

    plt.figure()
    plot_confusion_matrix(conf_matrix_custom, classes=set(merge_classes(custom_mapping=custom_mappings())),  normalize=False, title='Confusion matrix, without normalization')
    plt.show()

    conf_matrix_custom[0, 0] = 0

    log("accuracy_score no bg", np.sum([conf_matrix_custom[0, 0], conf_matrix_custom[1,1], conf_matrix_custom[2,2] ,conf_matrix_custom[3,3]]) / np.sum(conf_matrix_custom))
    log("balanced_accuracy_score no bg", balanced(conf_matrix_custom))

    plt.figure()
    plot_confusion_matrix(conf_matrix_custom, classes=set(merge_classes(custom_mapping=custom_mappings())), normalize=False, title='Confusion matrix, without normalization')
    plt.show()

    if return_metrics:
        return result, conf_matrix, conf_matrix_custom
    else:
        return result


def transform_masks_to_superpixel_summary(results):

    final_results = []
 
    for result in results:
        final_results.append(transform_masks_to_superpixel(result[0], result[1], result[2], True))

    return final_results


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


def display_colored_instances(original_image, boxes, masks, class_ids, class_names, scores=None):
    
    colors = {
        "cat_1": (0, 0.5, 0),
        "cat_2": (0, 1, 1),
        "cat_3": (1, 0, 1),
        "cat_4": (0, 0, 1),
        "cat_5": (1, 0, 0),
        "cat_6": (0, 0, 0)
    }

    final_colors = list()
    for classs in class_ids:
        final_colors.append(colors['cat_' + str(classs)])

    visualize.display_instances(image=original_image, masks=masks, boxes=boxes, class_ids=class_ids,
                                class_names=class_names, scores=scores, figsize=(8, 8),  colors=final_colors)
    
    log('rois', boxes)
    log('masks', masks)
    log('class_ids', class_ids)
    log('scores', scores)
