import os
import sys
import random
import numpy as np
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from pycocotools.coco import COCO
import tensorflow as tf
from preprocess_coco import process_data
from tooth import ToothConfig, ToothDataset
import matplotlib
matplotlib.use('tkagg')
import os
import sys
import random
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from operator import itemgetter 
from itertools import groupby


def get_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def create_superpixels(image=None, image_file="C:\\Projects\\tooth_damage_detection_deeplab\\data\\annotator\\training\\anaxristina37.jpg"):
    
    if(not image.any()):
        # load the image and convert it to a floating point data type
        image = img_as_float(io.imread(image_file))
    
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    #return slic(image, n_segments = 9, max_iter=9, sigma = .15)#.astype(np.uint8)

    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt

    segments = slic(image, n_segments = 900, max_iter=9, sigma = .15)#.astype(np.uint8)
            
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (9))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    io.imsave('C:\\Projects\\tooth_damage_detection\\src\\tests\\superpixels.jpg', mark_boundaries(image, segments))
    plt.axis("off")

    # show the plots
    plt.show()
    return segments


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    io.imsave('C:\\Projects\\tooth_damage_detection\\src\\tests\\masked_image.jpg', masked_image.astype(np.uint8))

    ax.imshow(masked_image.astype(np.uint8))
    
    if auto_show:
        plt.show()


def combine_masks_and_superpixels(masks, class_ids, superpixels):
    exit()
    n_superpixels = superpixels[-1, -1]
    print('Number of superpixels', n_superpixels)
    print('Superpixels', superpixels.shape)
    
    print('Number of masks', masks.shape[2])
    print('Masks', masks.shape)
    
    result = []
    result_class = []
    number_of = 0
    
    for i in range(n_superpixels):
        superpixel = superpixels == i
        # print(superpixels[np.where(superpixels == i)])
        # exit()
        print('Superpixel: ' + str(i) + ' / ' + str(n_superpixels), superpixel)
        # print('Superpixel volume: ', get_superpixel_volume(superpixel, superpixels, i))
        print('Superpixel shape: ', superpixel.shape)

        superpixel_classes = get_superpixel_classes(superpixel, masks, class_ids)

        print('Superpixel classes found', superpixel_classes)

        if(len(superpixel_classes) > 0):
            final_class = decide_class(superpixel_classes)

            print('Superpixel: ' + str(i) + ' / ' + str(n_superpixels) + ' class seleted', final_class)
            exit()

            result.append(superpixel)
            result.append(superpixel)
            result_class.append(final_class[0])
            number_of = number_of + 1
    
    print(masks)
    print(type(masks))
    
    return result, result_class, number_of


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
        #     print(np.where(superpixel[x] == True))
        #     print(np.unique(superpixel[x]))

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
        print('Searhing for mask with index: ' + str(mask_index) + ' with class: ' + str(class_id))
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

    # superpixel_temp = np.array([[1 if x == True else 0 for x in y] for y in superpixel])
    # mask_temp = np.array([[1 if x == True else 2 for x in y] for y in mask])

    result = []
    for p_x in range(superpixel_temp.shape[0]):

        temp_reult = superpixel_temp[p_x] == mask_temp[p_x]
        line_result = temp_reult[np.where(temp_reult == True)].shape[0]
        
        if (line_result > 0):

            print('Searhing in line with index (' + str(p_x) + '): ', superpixel_temp[p_x])
            print('For in mask: ', mask_temp[p_x])
            print('Result Indices: ', np.where(temp_reult == True))
            print('Result: ', temp_reult[np.where(temp_reult == True)])

        result.append(line_result)

    return np.sum(result)


data_dir = 'C:/Projects/tooth_damage_detection/data/'
training_data_dir = data_dir + 'output/training/'
validation_data_dir = data_dir + 'output/validation/'
unknown_data_dir = data_dir + 'output/unknown/'
annotation_file = '_annotation_data.json'

MODEL_DIR = "C:/Users/filippisc/Desktop/master/new_tests/final_test_noresize//"


print("Loading training dataset")
# Training dataset
dataset_train = ToothDataset()
dataset_train.load_data(training_data_dir)
dataset_train.prepare()

print("Loading validation dataset")
# Validation dataset
dataset_val = ToothDataset()
dataset_val.load_data(validation_data_dir)
dataset_val.prepare()

print("Loading unknown dataset")
# unknown dataset
dataset_unknown = ToothDataset()
dataset_unknown.load_data(unknown_data_dir)
dataset_unknown.prepare()


class InferenceConfig(ToothConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# Test on a random image
image_id = 3 #random.choice(dataset_unknown.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_unknown, inference_config,
                           image_id, use_mini_mask=False)

io.imsave('C:\\Projects\\tooth_damage_detection\\src\\tests\\image.jpg', original_image)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
print("image_id: ", image_id)


# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                             dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=0)


r = results[0]

colors = {

    "cat_1": (0, 0.5, 0),
    "cat_2": (0, 1, 1),
    "cat_3": (1, 0, 1),
    "cat_4": (0, 0, 1),
    "cat_5": (1, 0, 0),
    "cat_6": (0, 0, 0)
}

final_colors = list()
for classs in r['class_ids']:
    final_colors.append(colors['cat_' + str(classs)])

display_instances(image=original_image, masks=r["masks"], boxes=r['rois'], class_ids=r["class_ids"],
                            class_names=dataset_val.class_names, figsize=(8, 8), colors=final_colors)

msks, cls_ids = combine_masks_and_superpixels(r['masks'].astype(np.uint8), r['class_ids'].astype(np.uint8), create_superpixels(image=original_image))

display_instances(image=original_image, masks=np.array(msks), boxes=r['rois'], class_ids=np.array(cls_ids),
                            class_names=dataset_val.class_names, figsize=(8, 8), colors=final_colors)

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_unknown.image_ids, 4)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_unknown, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))