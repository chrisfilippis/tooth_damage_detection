from read_roi import read_roi_zip
from os import listdir
import os
from os.path import isfile, join
import json
from shutil import copyfile


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
    annotation_polygon = read_roi_zip(annotation_file_path)
    roi = list(annotation_polygon.items())
    polygons = get_polygons(roi)
    return [poly for poly in polygons if filter_clause not in poly[0]]


def process_regions_of_interest(roi_files):
    regions = []

    for region in roi_files:
        dental_class = int(region[0].split('-')[0])
        regions.append((dental_class, region[1], region[2]))

    return regions


def create_image_json_data_for_image(image_name, name_mapping_dic):
    return {
        "id": name_mapping_dic[image_name],
        "width": 1024,
        "height": 768,
        "file_name": image_name,
        "date_captured": "2013-11-15 02:41:42"
    }


def create_annotation_json_data_for_image(image_name, regions, name_mapping_dic):
    regions_data = []
    i = 0
    for region in regions:
        annot = []

        for ii in range(0, len(region[1])):
            annot.append(int(region[1][ii]))
            annot.append(int(region[2][ii]))

        annotation_data = {
            "id": image_name + '_' + str(i),
            "category_id": i,
            "iscrowd": 0,
            "segmentation": [annot],
            "image_id": name_mapping_dic[image_name],
        }

        regions_data.append(annotation_data)
        i += 1

    return regions_data


def create_final_coco_json(image_json, annotation_json):
    return {
        "images": image_json,
        "annotations": annotation_json[0],
        "info": {
            "description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01"
        },
        "categories": [
            {"id": 0, "name": "cat_0"},
            {"id": 1, "name": "cat_1"},
            {"id": 2, "name": "cat_2"},
            {"id": 3, "name": "cat_3"},
            {"id": 4, "name": "cat_4"},
            {"id": 5, "name": "cat_5"}
        ]
    }


def get_image_name(zipfile_name, name_mapping_dict):
    file_parts = zipfile_name.split('_')[1:3]
    name = file_parts[0] + '.' + file_parts[1]

    name = zipfile_name.replace('ANN_', '').split('_jpg')[0] + '.jpg'

    if name not in name_mapping_dict:
        name_mapping_dict[name] = len(name_mapping_dict)

    return name, name_mapping_dict


def empty_directory(directory_path):
    for f in [f for f in os.listdir(directory_path)]:
        os.remove(os.path.join(directory_path, f))


def process_data(input_directory, output_directory, annotation_file_name='region_data.json'):
    # input_directory = 'C:\\Projects\\tooth_damage_detection\\data\\annotator\\'
    # images_directory = 'C:\\Projects\\tooth_damage_detection\\data\\annotator\\'
    # output_directory = 'C:\\Projects\\tooth_damage_detection\\data\\output\\'

    if os.path.isfile(output_directory + annotation_file_name):
        return;

    empty_directory(output_directory)
    annotation_files = get_input_files(input_directory)

    final_images_data = []
    final_annotations_data = []

    name_mapping = dict();

    for annotation_filename in annotation_files:
        print('opening... ' + annotation_filename)

        image_name, name_mappings = get_image_name(annotation_filename, name_mapping)
        name_mapping = name_mappings
        annotation_file_path = input_directory + annotation_filename

        try:
            polygons = get_roi_files_from_zipfile(annotation_file_path, 'superpixel')
        except:
            continue

        regions = process_regions_of_interest(polygons)
        print(str(len(regions)) + ' regions found')

        copyfile(input_directory + image_name, output_directory + image_name)

        final_images_data.append(create_image_json_data_for_image(image_name, name_mapping))
        final_annotations_data.append(create_annotation_json_data_for_image(image_name, regions, name_mapping))

    with open(output_directory + annotation_file_name, 'w') as outfile:
        json.dump(create_final_coco_json(final_images_data, final_annotations_data), outfile)
