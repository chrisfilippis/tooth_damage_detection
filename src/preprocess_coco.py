from read_roi import read_roi_zip, read_roi_file
from os import listdir
import os
from os.path import isfile, join
import json
from shutil import copyfile
import zipfile
import numpy as np
import operator as operator
from matplotlib.path import Path
from skimage import draw
from skimage.draw import polygon
import skimage



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

def polygon2mask(image_shape, polygon):
    """Compute a mask from polygon.
    Parameters
    ----------
    image_shape : tuple of size 2.
        The shape of the mask.
    polygon : array_like.
        The polygon coordinates of shape (N, 2) where N is
        the number of points.
    Returns
    -------
    mask : 2-D ndarray of type 'bool'.
        The mask that corresponds to the input polygon.
    Notes
    -----
    This function does not do any border checking, so that all
    the vertices need to be within the given shape.
    Examples
    --------
    >>> image_shape = (128, 128)
    >>> polygon = np.array([[60, 100], [100, 40], [40, 40]])
    >>> mask = polygon2mask(image_shape, polygon)
    >>> mask.shape
    (128, 128)
    """
    polygon = np.asarray(polygon)
    vertex_row_coords, vertex_col_coords = polygon.T
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, image_shape)
    mask = np.zeros(image_shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def get_input_files(directory_path, extension='zip'):
    return [f for f in listdir(directory_path) if isfile(join(directory_path, f)) and f.endswith('.' + extension)]


def get_polygons(roi_file):
    return [get_polygon_info(polygon) for polygon in roi_file]


def get_polygon_info(polygon):
    name = polygon[1]['name']
    x = polygon[1]['x']
    y = polygon[1]['y']
    return name, x, y


def get_roi_files_from_zipfile(annotation_file_path, filter_clause='superpixel'):
    annotation_polygon = filter_zip(annotation_file_path)
    roi = list(annotation_polygon.items())
    polygons = get_polygons(roi)
    return [poly for poly in polygons if filter_clause not in poly[0]]


def filter_zip(zip_path):
    from collections import OrderedDict
    rois = OrderedDict()
    zf = zipfile.ZipFile(zip_path)
    for n in zf.namelist():
        if n.endswith('.roi'):
            rois.update(read_roi_file(zf.open(n)))
    return rois


def process_regions_of_interest(roi_files):
    regions = []

    for region in roi_files:
        dental_class = int(region[0].split('-')[0])

        if dental_class == 0:
            continue

        if dental_class > 6:
            dental_class = 6

        regions.append((dental_class, region[1], region[2]))

    return regions


def sort_cordinates(x, y):
    L = sorted(zip(x,y), key=operator.itemgetter(0))
    return zip(*L)


def get_class_from_roi(s):
    try: 
        return int(s.split('-')[0])
    except ValueError:
        return 0


def get_superpixels_new(annotation_file_path):
    annotation_polygon = filter_zip(annotation_file_path)
    roi = list(annotation_polygon.items())
    polygons = get_polygons(roi)

    superpixels = [poly for poly in polygons if get_class_from_roi(poly[0]) == 0]
    annotations = [poly for poly in polygons if get_class_from_roi(poly[0]) > 0]
    
    final_superpixels = []

    for annotation in annotations:
        final_superpixels.append((int(annotation[0].split('-')[0] ), np.array(annotation[1]).astype(np.int32), np.array(annotation[2]).astype(np.int32)))

    for superpixel in superpixels:
        final_superpixels.append((0, np.array(superpixel[1]).astype(np.int32), np.array(superpixel[2]).astype(np.int32)))

    result = np.zeros((768, 1024))
    result_classes = []

    ii = 1
    print('annotations found', len(annotations))
    print('superpixels found', len(superpixels))    
    print('final superpixels found', len(final_superpixels))


    # image = np.zeros((128, 128))
    # image_shape = image.shape
    # polygon = np.array([[1, 1], [1, 127], [127, 127], [127, 1]])
    # mask = polygon2mask(image_shape, polygon)
    # image[mask] = 1
    # print(image.shape)
    # print(mask)
    # exit()
    
    for final_superpixel in final_superpixels:
        dd = []

        for i in range(len(final_superpixel[1])):
            f_y = final_superpixel[2][i]
            f_x = final_superpixel[1][i]
            if(f_x == 1023):
                f_x +=1
            if(f_y == 767):
                f_y +=1

            dd.append([f_y, f_x])
        
        # print(dd)
        # polyg = np.array((final_superpixel[1], final_superpixel[2])).tolist()
        mask = polygon2mask(result.shape, dd)
        result[mask] = ii
        ii = ii + 1
        result_classes.append(final_superpixel[0])
        # print(ii)
        # print('result[mask]', result[mask])
        # print(result)

    print('result min', min(min(x) for x in result))
    print('result', max(max(x) for x in result))
    print('result_classes', len(result_classes))
    print('result result', sum(result[result == 0]))
    
    # for row in np.where(result == 0):
    #     print(row)

    # exit()

    return result.astype(np.int32), result_classes

### rewrite to be more efficient
def get_superpixels(annotation_file_path):

    annotation_polygon = filter_zip(annotation_file_path)
    roi = list(annotation_polygon.items())
    polygons = get_polygons(roi)

    superpixels = [poly for poly in polygons if get_class_from_roi(poly[0]) == 0]
    annotations = [poly for poly in polygons if get_class_from_roi(poly[0]) > 0]
    
    final_superpixels = []

    for annotation in annotations:
        final_superpixels.append((int(annotation[0].split('-')[0] ), np.array(annotation[1]).astype(np.int32), np.array(annotation[2]).astype(np.int32)))

    for superpixel in superpixels:
        final_superpixels.append((0, np.array(superpixel[1]).astype(np.int32), np.array(superpixel[2]).astype(np.int32)))

    result = np.zeros((768, 1024))
    result_classes = []

    x, y = np.meshgrid(np.arange(1024), np.arange(768))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T

    ii = 1
    print('annotations found', len(annotations))
    print('superpixels found', len(superpixels))    
    print('final superpixels found', len(final_superpixels))
    
    for final_superpixel in final_superpixels:
        poly_verts = []
        for i in range(len(final_superpixel[1])):
            poly_verts.append((final_superpixel[1][i], final_superpixel[2][i]))

        path = Path(poly_verts)
        grid = path.contains_points(points)
        grid = grid.reshape((768, 1024))
        result[grid] = ii
        result_classes.append(final_superpixel[0])
        ii = ii + 1
        # print(ii)

    print(result)

    return result, result_classes


def create_image_json_data_for_image(image_name, name_mapping_dic, annotation_file_path):
    return {
        "id": name_mapping_dic[image_name],
        "width": 1024,
        "height": 768,
        "file_name": image_name,
        "date_captured": "2013-11-15 02:41:42",
        "annotation_file_path": annotation_file_path
    }


def create_annotation_json_data_for_image(image_name, regions, name_mapping_dic):
    regions_data = []
    i = 0
    for region in regions:
        annot = []

        for ii in range(0, len(region[1])):
            annot.append((region[1][ii]))
            annot.append((region[2][ii]))

        annotation_data = {
            "id": (name_mapping_dic[image_name] * 10000) + i,
            "category_id": region[0],
            "iscrowd": 0,
            "segmentation": [annot],
            "image_id": name_mapping_dic[image_name],
        }

        regions_data.append(annotation_data)
        i += 1

    return regions_data


def create_final_coco_json(image_json, annotation_json):

    final_annotation_json = []

    for ii in annotation_json:
        for i in ii:
            final_annotation_json.append(i)
    return {
        "images": image_json,
        "annotations": final_annotation_json,
        "info": {
            "description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01"
        },
        "categories": [
            {"id": 1, "name": "cat_1"},
            {"id": 2, "name": "cat_2"},
            {"id": 3, "name": "cat_3"},
            {"id": 4, "name": "cat_4"},
            {"id": 5, "name": "cat_5"},
            {"id": 6, "name": "cat_6"}
        ]
    }


def get_image_name(zipfile_name, name_mapping_dict):
    file_parts = zipfile_name.split('_')[1:3]
    name = file_parts[0] + '.' + file_parts[1]

    name = zipfile_name.replace('ANN_', '').split('_jpg')[0] + '.jpg'

    if name not in name_mapping_dict:
        name_mapping_dict[name] = len(name_mapping_dict) + 1

    return name, name_mapping_dict


def ensure_directory_existence(directory_path):
    if not os.path.isdir(directory_path):
        try:
            os.mkdir(directory_path)
        except OSError:
            print("Creation of the directory %s failed" % directory_path)
        return


def empty_directory(directory_path):
    for f in [f for f in os.listdir(directory_path)]:
        os.remove(os.path.join(directory_path, f))


def process_data(input_directory, output_directory, annotation_file_name='region_data.json', force_load=False):
    # input_directory = 'C:\\Projects\\tooth_damage_detection\\data\\annotator\\'
    # images_directory = 'C:\\Projects\\tooth_damage_detection\\data\\annotator\\'
    # output_directory = 'C:\\Projects\\tooth_damage_detection\\data\\output\\'

    if not force_load and os.path.isfile(output_directory + annotation_file_name):
        return

    ensure_directory_existence(output_directory)
    empty_directory(output_directory)
    annotation_files = get_input_files(input_directory)

    final_images_data = []
    final_annotations_data = []

    name_mapping = dict()

    for annotation_filename in annotation_files:
        print('opening... ' + annotation_filename)

        image_name, name_mappings = get_image_name(annotation_filename, name_mapping)
        name_mapping = name_mappings
        annotation_file_path = input_directory + annotation_filename

        polygons = get_roi_files_from_zipfile(annotation_file_path, 'superpixel')

        regions = process_regions_of_interest(polygons)
        print(str(len(regions)) + ' regions found')

        copyfile(input_directory + image_name, output_directory + image_name)

        final_images_data.append(create_image_json_data_for_image(image_name, name_mapping, input_directory + annotation_filename))

        image_json = create_annotation_json_data_for_image(image_name, regions, name_mapping)
        final_annotations_data.append(image_json)

    with open(output_directory + annotation_file_name, 'w') as outfile:
        file_json = create_final_coco_json(final_images_data, final_annotations_data)
        json.dump(file_json, outfile)
