from read_roi import read_roi_zip, read_roi_file
from os import listdir
from os.path import isfile, join


def get_input_files(directory_path, extension='zip'):
    return [f for f in listdir(directory_path) if isfile(join(directory_path, f)) and f.endswith('.' + extension)]


def get_polygons(roi_file):
    return [get_polygon_info(polygon) for polygon in roi_file]


def get_polygon_info(polygon):
    name = polygon[1]['name']
    x = polygon[1]['x']
    y = polygon[1]['y']
    return name, x, y


def get_superpixels_from_zipfile(annotation_file_path):
    annotation_polygon = read_roi_zip(annotation_file_path)
    roi = list(annotation_polygon.items())
    polygons = get_polygons(roi)
    return [poly for poly in polygons if 'superpixel' in poly]


def main():
    input_directory = 'C:\\Projects\\tooth_damage_detection\\data\\annotator\\'
    annotation_files = get_input_files(input_directory)

    for annotation_filename in annotation_files:
        print(annotation_filename)
        annotation_file_path = input_directory + annotation_filename
        polygons = get_superpixels_from_zipfile(annotation_file_path)
        print(polygons)


if __name__ == "__main__":
    main()