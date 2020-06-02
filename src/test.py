# from customize import create_superpixels
import numpy as np
import glob, os
from preprocess_coco import get_superpixels, get_superpixels_new
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries


def find_file(directory, image):
    pattern = '*' + image.replace('.', '_') + '*.zip'
    print(pattern)
    os.chdir(directory)
    for file in glob.glob(pattern):
        return file

# dd = get_superpixels('C:\Projects\\tooth_damage_detection\data\\annotator\\training\\ANN_antanastasios36_jpg_2020-05-31T15_58_22_506216600Z.zip')
# dd = get_superpixels_new("C:\\Projects\\tooth_damage_detection\data\\annotator/validation/ANN_geotsampikos36_jpg_2018-12-09T19_12_28_786Z.zip")
segments,s = get_superpixels_new("C:\\Projects\\tooth_damage_detection\\data\\annotator\\validation\\ANN_anaxristina37_2_jpg_2018-12-09T20_43_04_849Z.zip")
# print(segments.astype(int))
original_image = img_as_float(io.imread('C:\\Projects\\tooth_damage_detection\\data\\annotator\\validation\\anaxristina37_2.jpg'))
# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (9))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(original_image, segments.astype(int)))
plt.axis("off")

# show the plots
plt.show()


# print(merge_classes([0,1,1,3,4,5,6]))

# print(merge_classes([1,2,3,4,5,6], {1:1, 2:1, 3:1, 4:2, 5:1, 6:1, 0:1}))
# print(merge_classes())

# print(create_superpixels())