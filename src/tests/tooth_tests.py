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
from src.preprocess_coco import process_data
from src.tooth import ToothConfig, ToothDataset
import matplotlib
matplotlib.use('tkagg')

# Directory to save logs and trained model

data_dir = 'C:/Projects/tooth_damage_detection/data/'

training_data_dir = data_dir + 'output/training/'
validation_data_dir = data_dir + 'output/validation/'
unknown_data_dir = data_dir + 'output/unknown/'

annotation_file = '_annotation_data.json'

force_load = False
process_data(data_dir + 'annotator/training/', training_data_dir, annotation_file, force_load=force_load)
process_data(data_dir + 'annotator/validation/', validation_data_dir, annotation_file, force_load=force_load)
process_data(data_dir + 'annotator/unknown/', unknown_data_dir, annotation_file, force_load=force_load)

MODEL_DIR = "C:/Users/filippisc/Desktop/master/tests//"

training_data_dir = data_dir + 'output/training'
unknown_data_dir = data_dir + 'output/unknown'

annotation_file = '_annotation_data.json'

config = ToothConfig()

print("Loading training dataset")
# Training dataset
dataset_train = ToothDataset()
dataset_train.load_data(training_data_dir)
dataset_train.prepare()

print("Loading validation dataset")
# Validation dataset
dataset_unknown = ToothDataset()
dataset_unknown.load_data(unknown_data_dir)
dataset_unknown.prepare()

print(dataset_train.class_ids)
print(dataset_train.class_names)
print(dataset_train.class_from_source_map)

inference_config = ToothConfig()

model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
# visualize.display_weight_stats(model)

# Test on a random image
print(dataset_unknown.image_ids)
img_id = dataset_unknown.image_ids[2]

print("img_id ", img_id)

original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modellib.load_image_gt(dataset_unknown, inference_config,
                           img_id, use_mini_mask=False)

# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)


# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                             dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
print(len(r['rois']))
# print(r['class_ids'])
# print(r['masks'])
# print(r['rois'])

visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_unknown.class_names, figsize=(8, 8))

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_unknown.image_ids, 10)
APs = []
for img_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_unknown, inference_config,
                               img_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
    #                             dataset_unknown.class_names, r['scores'], ax=get_ax())

    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])

    APs.append(AP)

print("mAP: ", np.mean(APs))
