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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
    n_superpixels = int(n_superpixels)
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
        print('Superpixels: ', superpixels)
        print('Superpixel: ', superpixel)
        print('Superpixel volume: ', get_superpixel_volume(superpixel, superpixels, i))
        superpixel_classes = get_superpixel_classes(superpixel, masks, class_ids)

        # print('Superpixel classes found', superpixel_classes)

        if(len(superpixel_classes) > 0):
            final_class = decide_class(superpixel_classes)
            
            print('Superpixel ' + str(i) + ' class found ' +  str(final_class[0]))
        
            superpixels_classes.append(final_class[0])
            superpixel_bbox = get_bbox(superpixel)
            # print('Superpixel volume: ', get_superpixel_volume(superpixel, superpixels, i))
            # print('superpixel_bbox: ', superpixel_bbox)
            superpixels_bboxes.append(superpixel_bbox)
        else:
            # print('Superpixel ' + str(i) + ' class not found')
            superpixels_classes.append(100)    

    print('n_superpixels', n_superpixels)
    print('test_superpixels_classes_1', len(superpixels_classes))
    
    superpixels_classes = np.array(superpixels_classes)

    print('test_superpixels_classes', len(superpixels_classes))

    classes = np.unique(superpixels_classes[superpixels_classes != 100])
    classes = superpixels_classes[superpixels_classes != 100]

    result_masks = np.zeros((superpixels.shape[0], superpixels.shape[1], classes.shape[0]))

    # loop superpixels verticaly
    for i in range(n_superpixels):
        superpixel_class = superpixels_classes[i]
        
        if(superpixel_class == 100):
            continue

        superpixel = superpixels == i
    
        sup_class = classes == superpixel_class
        # sup_class = sup_class.astype(np.uint8)

        result_masks[superpixel] = sup_class

    return [{
        'class_ids': classes,
        'masks': result_masks.astype(np.bool),
        'rois': np.array(superpixels_bboxes).astype(np.int32),
        'scores' : np.full((result_masks.shape[2],), 0.21).astype(np.float32),
        'superpixels_classes': superpixels_classes.astype(np.int32)
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
      
        mask[mask == 1] = True
        mask[mask == 0] = False

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


def transform_masks_to_superpixel(results, original_image, original_image_annotation):

    print('annotation_file', original_image_annotation)

    r = results[0]
    # superpixels = create_superpixels(image=original_image)
    print('loading superpixels...')
    superpixels, superpixels_classes = get_superpixels_new(original_image_annotation)
    
    print('superpixels found.', superpixels.shape)
    print(str(len(superpixels_classes)) + ' superpixels classes found.')

    print('combine_masks_and_superpixels...')
    result = combine_masks_and_superpixels(r['masks'].astype(np.uint8), r['class_ids'].astype(np.uint8), superpixels)
    predictions = result[0]['superpixels_classes']
    predictions[predictions == 100] = 0
    
    print('superpixels_classes', len(superpixels_classes))
    print('result', predictions)
    print('result', len(predictions))

    predictions[predictions == 100] = 0
    
    predictions = merge_classes(classes=predictions, custom_mapping=None)
    superpixels_classes = merge_classes(classes=superpixels_classes, custom_mapping=None)

    conf_matrix = confusion_matrix(superpixels_classes, predictions, labels=merge_classes())

    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=set(merge_classes()), normalize=False, title='Confusion matrix, without normalization')
    plt.show()

    custom_mapping = {
        0:0,
        1:1,
        2:1,
        3:2,
        4:3,
        5:3,
        6:3 }

    predictions = merge_classes(classes=predictions, custom_mapping=custom_mapping)
    superpixels_classes = merge_classes(classes=superpixels_classes, custom_mapping=custom_mapping)

    conf_matrix = confusion_matrix(superpixels_classes, predictions, labels=merge_classes(custom_mapping=custom_mapping))

    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=set(merge_classes(custom_mapping=custom_mapping)), normalize=False, title='Confusion matrix, without normalization')
    plt.show()

    return result


def merge_classes(classes=None, custom_mapping=None):

    init_mapping = {
        0:0,
        1:1,
        2:2,
        3:3,
        4:4,
        5:5,
        6:6 }

    mapping = init_mapping

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
