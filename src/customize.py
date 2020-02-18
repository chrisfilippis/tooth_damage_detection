import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from operator import itemgetter 
from itertools import groupby
from mrcnn import visualize


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
    n_superpixels = superpixels[-1, -1]
    # print('Number of superpixels', n_superpixels)
    # print('Superpixels', superpixels.shape)
    
    # print('Number of masks', masks.shape[2])
    # print('Masks', masks.shape)

    superpixels_classes = []
    superpixels_bboxes = []
    # loop superpixels verticaly
    for i in range(n_superpixels):
        superpixel = superpixels == i
        superpixel_classes = get_superpixel_classes(superpixel, masks, class_ids)

        if(len(superpixel_classes) > 0):
            final_class = decide_class(superpixel_classes)

            # print('Superpixel classes found', superpixel_classes)
            # print('Superpixel: ' + str(i) + ' / ' + str(n_superpixels) + ' class seleted', final_class)

            # print('superpixel.shape', superpixel.shape)
            # print('Superpixel shape: ', superpixel.shape)
            
            superpixels_classes.append(final_class[0])
            superpixel_bbox = get_bbox(superpixel)
            # print('Superpixel volume: ', get_superpixel_volume(superpixel, superpixels, i))
            # print('superpixel_bbox: ', superpixel_bbox)
            superpixels_bboxes.append(superpixel_bbox)
        else:
            superpixels_classes.append(100)
    
    superpixels_classes = np.array(superpixels_classes)

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
        'scores' : np.full((result_masks.shape[2],), 0.21).astype(np.float32)
    }]


def decide_class(superpixel_classes):
    result = []
    for key, temp in groupby(superpixel_classes, key = itemgetter(1)):
        result.append((key, sum(list(map(itemgetter(0), temp)))))
    
    return sorted(result, key=lambda tup: tup[1])[-1]


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


def get_superpixel_classes(superpixel, masks, class_ids):

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


def transform_masks_to_superpixel(results, original_image, show_image=False):

    r = results[0]
    superpixels = create_superpixels(image=original_image)
    return combine_masks_and_superpixels(r['masks'].astype(np.uint8), r['class_ids'].astype(np.uint8), superpixels)


def display_colored_instances(results, original_image, class_names):
    colors = {
        "cat_1": (0, 0.5, 0),
        "cat_2": (0, 1, 1),
        "cat_3": (1, 0, 1),
        "cat_4": (0, 0, 1),
        "cat_5": (1, 0, 0),
        "cat_6": (0, 0, 0)
    }

    r = results[0]

    final_colors = list()
    for classs in r['class_ids']:
        final_colors.append(colors['cat_' + str(classs)])

    visualize.display_instances(image=original_image, masks=r["masks"], boxes=r['rois'], class_ids=r["class_ids"],
                                class_names=class_names, scores=r["scores"], figsize=(8, 8),  colors=final_colors)
