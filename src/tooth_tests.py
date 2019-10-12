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


data_dir = 'C:/Projects/tooth_damage_detection/data/'

training_data_dir = data_dir + 'output/training/'
validation_data_dir = data_dir + 'output/validation/'
unknown_data_dir = data_dir + 'output/unknown/'

annotation_file = '_annotation_data.json'

MODEL_DIR = "C:/Users/filippisc/Desktop/master/tests//heads_20_decay00005_anchors_2//"


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
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], figsize=(8, 8))

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
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