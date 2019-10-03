import os
import random
import numpy as np
import tensorflow as tf
from pycocotools import mask as maskUtils
import imgaug

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from pycocotools.coco import COCO
from preprocess_coco import process_data


class ToothConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 3 shapes


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

        coco = COCO(dataset_dir + "\\" + annotation_filename)

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


def main():
    # Directory to save logs and trained model

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')

    parser.add_argument('--data_dir', required=False, help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--model_dir', required=False, help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--init_with', required=False, default="coco", help="imagenet, coco, or last")

    args = parser.parse_args()

    print("data_dir: ", args.data_dir)
    print("model_dir: ", args.model_dir)
    print("init_with: ", args.init_with)

    data_dir = 'C:\\Projects\\tooth_damage_detection\\data\\'
    MODEL_DIR = os.path.join(data_dir, "logs")

    if args.data_dir is not None:
        data_dir = args.data_dir

    if args.model_dir is not None:
        MODEL_DIR = args.model_dir

    model_file = "mask_rcnn_coco.h5"
    DEVICE = '/gpu:1'  # /gpu:1  /cpu:0 or /gpu:0

    training_data_dir = data_dir + 'output/training/'
    validation_data_dir = data_dir + 'output/validation/'
    unknown_data_dir = data_dir + 'output/unknown/'

    annotation_file = '_annotation_data.json'

    # Which weights to start with?
    init_with = args.init_with  # imagenet, coco, or last

    force_load = False
    process_data(data_dir + 'annotator/training/', training_data_dir, annotation_file, force_load=force_load)
    process_data(data_dir + 'annotator/validation/', validation_data_dir, annotation_file, force_load=force_load)
    process_data(data_dir + 'annotator/unknown/', unknown_data_dir, annotation_file, force_load=force_load)

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(data_dir, model_file)
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

    with tf.device(DEVICE):
        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)

    print("load weights")

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    print("Train the head branches")

    augmentation = imgaug.augmenters.Fliplr(0.5)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    with tf.device(DEVICE):
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=20,
                    layers='heads',
                    augmentation=augmentation)

        print("Fine tune all layers")

    # # Fine tune all layers
    # # Passing layers="all" trains all layers. You can also
    # # pass a regular expression to select which layers to
    # # train by name pattern.
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=2,
    #             layers="all")

    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=60,
    #             layers='4+')

    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    # model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    # model.keras_model.save_weights(model_path)

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
    img_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               img_id, use_mini_mask=False)

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
    for img_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
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


if __name__ == '__main__':
    main()
