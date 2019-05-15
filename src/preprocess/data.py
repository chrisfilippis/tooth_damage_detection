from read_roi import read_roi_zip
from os import listdir
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


def create_json_data_for_image(image_name, regions):
    regions_data = {}
    i = 0
    for region in regions:
        region_data = {
                'shape_attributes': {
                    'name': 'polygon',
                    'all_points_x': [int(x) for x in region[1]],
                    'all_points_y': [int(y) for y in region[2]]
                },
                'region_attributes': {}
        }
        regions_data[str(i)] = region_data
        i += 1

    image_data = {
        'fileref': '',
        'filename': image_name,
        'base64_img_data': '',
        'file_attributes': {},
        'regions': regions_data
    }

    return image_data


def get_image_name(zipfile_name):
    file_parts = zipfile_name.split('_')[1:3]
    return file_parts[0] + '.' + file_parts[1]


def main():
    input_directory = 'C:\\Projects\\tooth_damage_detection\\data\\annotator\\'
    images_directory = 'C:\\Projects\\tooth_damage_detection\\data\\annotator\\'
    output_directory = 'C:\\Projects\\tooth_damage_detection\\data\\annotator\\output\\'
    annotation_files = get_input_files(input_directory)

    final_regions_data = {}

    for annotation_filename in annotation_files:
        print('opening... ' + annotation_filename)
        image_name = get_image_name(annotation_filename)
        annotation_file_path = input_directory + annotation_filename
        polygons = get_roi_files_from_zipfile(annotation_file_path, 'superpixel')

        regions = process_regions_of_interest(polygons)
        print(str(len(regions)) + ' regions found')

        copyfile(images_directory + image_name, output_directory + image_name)

        final_regions_data[image_name] = create_json_data_for_image(image_name, regions)

    with open(output_directory + "via_region_data.json", 'w') as outfile:
        json.dump(final_regions_data, outfile)


if __name__ == '__main__':
    main()
