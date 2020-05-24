import os
import random
import numpy as np
from pycocotools import mask as maskUtils
import imgaug
import imgaug.augmenters as iaa

from mrcnn.config import Config
import keras
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from pycocotools.coco import COCO
from preprocess_coco import process_data
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from customize import create_superpixels, combine_masks_and_superpixels, transform_masks_to_superpixel
import schedules


class ToothConfig(Config):

    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # (GPUs * images/GPU).
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6
    STEPS_PER_EPOCH = 57
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)  # anchor side in pixels
    
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1024
    IMAGE_RESIZE_MODE = "none"
    
    VALIDATION_STEPS = 9
    
    # TRAIN_ROIS_PER_IMAGE = 512
    # WEIGHT_DECAY = 0.0001


def train(model, data_train, data_val, cfg):
    schedules.schedule1(model, data_train, data_val, cfg)


class ToothDataset(utils.Dataset):
    def load_data(self, dataset_dir, annotation_filename='_annotation_data.json', return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        coco = COCO(dataset_dir + "//" + annotation_filename)

        # Load all classes or
        classes = sorted(coco.getCatIds())

        # All images or
        images = list(coco.imgs.keys())

        # Add classes
        for i in classes:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in images:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(dataset_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=classes, iscrowd=None)))
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        # if image_info["source"] != "coco":
        #     return super(ToothDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))

            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array

        if class_ids:
            mask_ = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask_, class_ids
        else:
            # Call super class to return an empty mask
            return super(ToothDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(ToothDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def visualize_colored_instances(model, class_names, img, results):
    colors = {
        "cat_1": (0, 0.5, 0),
        "cat_2": (0, 1, 1),
        "cat_3": (1, 0, 1),
        "cat_4": (0, 0, 1),
        "cat_5": (1, 0, 0),
        "cat_6": (0, 0, 0)
    }

    final_colors = find_colors(r['class_ids'])

    visualize.display_instances(img, results['rois'], results['masks'], results['class_ids'],
                                class_names, results['scores'], figsize=(8, 8))


def find_colors(class_ids):
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

    return final_colors


def measure_accuracy(MODEL_DIRECTORY, data_train, dat_val):
    class InferenceConfig(ToothConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIRECTORY)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Test on a random image
    img_id = random.choice(dat_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dat_val, inference_config,
                               img_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                data_train.class_names, figsize=(8, 8))

    colors = {
        "cat_1": (0, 0.5, 0),
        "cat_2": (0, 1, 1),
        "cat_3": (1, 0, 1),
        "cat_4": (0, 0, 1),
        "cat_5": (1, 0, 0),
        "cat_6": (0, 0, 0)
    }

    results = model.detect([original_image], verbose=1)
    r = results[0]

    final_colors = list()
    for classs in r['class_ids']:
        final_colors.append(colors['cat_' + str(classs)])

    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                data_train.class_names, r['scores'], figsize=(8, 8))

    msks, cls_ids, bboxes = transform_masks_to_superpixel(results, original_image, data_train.class_names)

    final_colors = list()
    for classs in cls_ids:
        final_colors.append(colors['cat_' + str(classs)])

    visualize.display_instances(image=original_image, masks=np.array(msks), boxes=np.array(bboxes), class_ids=np.array(cls_ids),
                            class_names=dataset_val.class_names, figsize=(8, 8), colors=final_colors)

    log('rois', r['rois'])
    log('masks', r['masks'][0][0])
    log('class_ids', r['class_ids'])
    log('scores', r['scores'])
    exit()

    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(data_train.image_ids, 10)
    APs = []
    for img_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(data_train, inference_config,
                                   img_id, use_mini_mask=False)
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


def main():

    # dataset_train = ToothDataset()
    # dataset_train.load_data("C:\\Projects\\tooth_damage_detection\data\\output\\training\\")
    # dataset_train.prepare()

    # dataset_val = ToothDataset()
    # dataset_val.load_data("C:\\Projects\\tooth_damage_detection\data\\output\\validation\\")
    # dataset_val.prepare()

    # measure_accuracy("C:\\Users\\filippisc\Desktop\master\\new_tests\\results\\test_1", dataset_train, dataset_val)
    # exit()

    # Directory to save logs and trained model

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train custom Mask R-CNN on MS COCO.')

    # parser.add_argument('--data_dir', required=False, help="Path to load data (it should has training, unkown, validation)")
    parser.add_argument('--model_file', required=False, default="mask_rcnn_coco.h5", help="Path to initial .h5 file")
    parser.add_argument('--init_with', required=False, default="coco", help="imagenet, coco, or last")
    parser.add_argument('--model_dir', required=False, help="Path to weights .h5 file or 'coco'")
    
    parser.add_argument('--input_training_data', default="/var/project/data/annotator/training/", required=False, help="input training data folder")
    parser.add_argument('--input_validation_data', default="/var/project/data/annotator/validation/",required=False, help="input validation data folder")
    parser.add_argument('--input_unknown_data', default="/var/project/data/annotator/unknown/", required=False, help="input unknown data folder")

    parser.add_argument('--training_data', default="/var/project/data/output/training/", required=False, help="final training data folder")
    parser.add_argument('--validation_data', default="/var/project/data/output/training/", required=False, help="final validation data folder")
    parser.add_argument('--unknown_data', default="/var/project/data/output/training/", required=False, help="final unknown data folder")

    args = parser.parse_args()

    # print("data_dir: ", args.data_dir)
    print("model_dir: ", args.model_dir)
    print("init_with: ", args.init_with)


    # if args.data_dir is not None:
    #     data_dir = args.data_dir
    # else:
    #     data_dir = 'C:\\Projects\\tooth_damage_detection\\data\\'

    if args.model_dir is not None:
        MODEL_DIR = args.model_dir
    else:
        MODEL_DIR ='C:\\Projects\\tooth_damage_detection\\data\\logs'


    if args.input_training_data is not None:
        input_training_data_dir = args.input_training_data
    else:
        input_training_data_dir = 'C:\\Projects\\tooth_damage_detection\\data\\annotator/training/'

    if args.input_validation_data is not None:
        input_validation_data_dir = args.input_validation_data
    else:
        input_validation_data_dir = 'C:\\Projects\\tooth_damage_detection\\data\\annotator/validation/'

    if args.input_unknown_data is not None:
        input_unknown_data_dir = args.input_unknown_data
    else:
        input_unknown_data_dir = 'C:\\Projects\\tooth_damage_detection\\data\\annotator/unknown/'

    
    if args.training_data is not None:
        training_data_dir = args.training_data
    else:
        training_data_dir = 'C:\\Projects\\tooth_damage_detection\\data\\output/training/'

    if args.validation_data is not None:
        validation_data_dir = args.validation_data
    else:
        validation_data_dir = 'C:\\Projects\\tooth_damage_detection\\data\\output/validation/'

    if args.unknown_data is not None:
        unknown_data_dir = args.unknown_data
    else:
        unknown_data_dir = 'C:\\Projects\\tooth_damage_detection\\data\\output/unknown/'

    if args.model_file is not None:
        model_file = args.model_file
        # COCO_MODEL_PATH = os.path.join(data_dir, model_file)
        COCO_MODEL_PATH = model_file
    else:
        COCO_MODEL_PATH = 'C:\\Projects\\tooth_damage_detection\\data\\mask_rcnn_coco.h5'

    annotation_file = '_annotation_data.json'

    # Which weights to start with?
    init_with = args.init_with  # imagenet, coco, or last

    force_load = False
    process_data(input_training_data_dir, training_data_dir, annotation_file, force_load=force_load)
    process_data(input_validation_data_dir, validation_data_dir, annotation_file, force_load=force_load)
    process_data(input_unknown_data_dir, unknown_data_dir, annotation_file, force_load=force_load)

    # Local path to trained weights file
    # COCO_MODEL_PATH = os.path.join(data_dir, model_file)
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        print("downloading")
        utils.download_trained_weights(COCO_MODEL_PATH)

    config = ToothConfig()
    config.display()

    print("Loading training dataset")
    # Training dataset
    dataset_train = ToothDataset()
    dataset_train.load_data(training_data_dir)
    dataset_train.prepare()

    print(dataset_train.class_names)
    print(dataset_train.class_ids)
    print(dataset_train.class_info)
    print(dataset_train.class_from_source_map)

    # Validation dataset
    dataset_val = ToothDataset()
    dataset_val.load_data(validation_data_dir)
    dataset_val.prepare()

    print("Create model in training mod")

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    print("load weights")

    # if init_with == "imagenet":
    #     model.load_weights(model.get_imagenet_weights(), by_name=True)
    # elif init_with == "coco":
    if init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        print(COCO_MODEL_PATH)
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    print("Train the head branches")

    train(model, dataset_train, dataset_val, config)

    print("Fine tune all layers")

    # measure_accuracy(MODEL_DIR, dataset_train, dataset_val)


if __name__ == '__main__':
    main()
