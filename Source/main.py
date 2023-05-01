'''
    Main code for automated tree tagging and analysis software
    Author: Daniel Butt, NTP 2022
    Email: dbutt7@uwo.ca
    Date of last major revision: Aug 26, 2022
'''
import time

import treetag
from treetag import *
import math
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import tensorflow as tf
from tqdm import tqdm
import timeit
import os
from datetime import datetime
from PIL import Image

np.seterr(all="ignore")
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# timer to measure total runtime
t = timeit.default_timer()

#checking if tensorflow was built with cuda support and the user has a compatible gpu
if tf.test.is_built_with_cuda():
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) != 0:
        print("CUDA ENABLED")
        print("Running on: ", devices[0])

#list of arguments passed from arcgis add in
argv = []
# path to arcgis project folder
project_path = ""

print("Loading arguments")
try:
    #geting project path
    argv = sys.argv[1:]
    project_path = argv[0]
    # project_path = r"F:\2022 Data\Aerial Surveys 2022\July 24 - Rockdale-Actinolite, ON\Rockdale-Actinolite"
except IndexError:
    print_error("Couldn't load Tree Tagger settings, make sure the entered settings are correct")

try:
    argv = load_args(project_path + "\\TreeTagger\\args.txt")
except FileNotFoundError:
    print_error("Couldn't load Tree Tagger arguments file, make sure the entered settings are correct")

#parameters for joining lines and finding high density areas
parameters = []

#interpolation type for building direction vectors
direction_interpolation = "W Average"
#image file paths and coordinates
images_data = []
#image file paths and coordinates for images bordering the images being processed
border_data = []
#the points of the polygon region selected, if a region was selected
polygon_points = []
#image scale (usually 5cm imagery so (0.05, 0.05))
image_scale = (0, 0)
#coordinate of bottom right corner
bottom_right = (0, 0)
#coordinate of top left corner
top_left = (0, 0)
#whether a polygon region was selected
is_polygon = False
#grid sizes for averaging direction vectors
direction_grid_parameters = []
#size to extend line for cropping section around a tree before assessing the trees direction
line_extension_size = 12.8
#Whether to perform a histogram equalization on each image's intensity (Drone Data mostly)
equalize = False

# loading and organizing all arguments passed from arcgis add in
# see add in code from the order of arguments
try:
    i = 0
    while argv[i] != "?":
        direction_grid_parameters.append([int(argv[i]), int(argv[i+1])])
        i += 2
    i += 1
    argv = argv[i:]

    if int(argv[0]) > 0:
        equalize = True

    argv = argv[1:]

    parameters = [float(x) for x in argv[:8]]

    argv = argv[8:]

    direction_interpolation = argv[0]

    image_scale = (round(float(argv[1]), 6), round(float(argv[2]), 6))

    top_left = (float(argv[3]), float(argv[4]))

    bottom_right = (float(argv[5]), float(argv[6]))

    is_polygon = bool(int(argv[7]))

    argv = argv[8:]

    x = 0
    if is_polygon:
        while is_float(argv[x]):
            polygon_points.append((float(argv[x]), float(argv[x+1])))
            x += 2

    argv = argv[x:]
    img_args = []
    border_args = []

    i = 0
    while i < len(argv) and "?" not in argv[i]:
        img_args.append(argv[i])
        i += 1

    i += 1

    while i < len(argv):
        border_args.append(argv[i])
        i += 1

    for i in range(0, len(img_args), 3):
        images_data.append((img_args[i], float(img_args[i + 1]), float(img_args[i + 2])))

    for i in range(0, len(border_args), 3):
        border_data.append((border_args[i], float(border_args[i + 1]), float(border_args[i + 2])))

except Exception:
    print_error("Couldn't parse Tree Tagger arguments file, make sure the entered settings are correct ")

#if a polygon region was selected, the size of the polygon is scaled down to save space when creating a mask
#to bitwise AND with each image being processed. The size of the scaled down mask depends on the amount of pixels
#which would encompass the bounding box the polygon (top_left -> bottom_right)
#The polygon is then resized to fit into an image with a fixed number of pixels
#
#total pixels in image = w * h, fixed pixel sized image = (original width/scale factor)*(original height/scale factor)
#scale factor = sqrt((original width * original height) / fixed pixel size)

region_width = (bottom_right[0] - top_left[0]) / image_scale[0]
region_height = (top_left[1] - bottom_right[1]) / image_scale[1]
region_scale = math.sqrt((region_width * region_height) / 1000000)

#make total pixels larger if bounding box is massive
if region_height >= 500000 or region_width >= 500000:
    region_scale = math.sqrt((region_width * region_height) / 1000000000)
elif region_height >= 50000 or region_width >= 50000:
    region_scale = math.sqrt((region_width * region_height) / 100000000)

# mask for polygon
mask = np.zeros([max(int(math.ceil(region_height / region_scale)), 1), max(int(math.ceil((region_width / region_scale))), 1)], np.uint8)

if region_width == 0 or region_height == 0 or region_scale == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
    print_error("invalid region selection, Region Width: " + str(region_width) + ", Region Height: " + str(region_height) +
               ", Region Scale: " + str(region_scale) + "Make sure you selected an image/polygon region which contains an image, also check that the coordinate systems match")

if is_polygon:
    print("Cropping out polygon region")
    #convert polygon points from coords to pixels and then resize the polygon to fit fixed scale factor
    try:
        #draw filled polygon on mask image so that it can be bitwise ANDed effectively
        crop_coord_polygon(polygon_points, image_scale, region_scale, top_left, mask)
    except Exception:
        print_error("Couldn't create mask for selected polygon, check that the coordinate systems match")


print("Loading machine learning model")
try:
    model = treetag.TreeSegModel(os.path.dirname(os.path.realpath(__file__)) + "\\weights4.h5")
except Exception:
    print_error("Couldn't load model, check Tree Tagger installation folder for weights file / Keras_Segmentation package")

print("Calculating predictions and detecting lines")
#all the lines detected between all images being processed
all_lines = []

for img_data in tqdm(images_data, file=sys.stdout):
    #offset to keep track of where lines are relative to the coordinates of the image being processed
    offset = ((img_data[1] - top_left[0]) / image_scale[0], (top_left[1] - img_data[2]) / image_scale[1])
    image = 0

    #loading the current image being processed and make sure its size and color bands (RGB and not gray scale, etc.)
    try:
        t = time.time()
        image = cv2.imread(img_data[0], cv2.IMREAD_COLOR)
        # print("\n", img_data[0], "time:", time.time()-t)
    except Exception:
        print_error("Couldn't load image: " + str(img_data[0]))

    if image.shape[0] < 512 or image.shape[1] < 512:
        continue

    if image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] != 3:
        print_error("invaild image: image width: " + str(image.shape[1]) + ", image height: " + str(image.shape[0]) + ", image color bands (rgb should be 3): " + str(image.shape[2]))

    #if a polygon region was selected, crop out and resize the fixed sized polygon mask so that it's the same size as
    #the image being processed. Then bitwise AND the image and mask to black out any region not contained in the polygon
    if is_polygon:
        try:
            image = extract_scaled_mask(image, mask, region_scale, offset)
        except Exception:
            print_error("Couldn't extract polygon mask, Make sure you selected an image/polygon region which contains an"
                       + " image, also check that the coordinate systems match, mask width: " +
                       str(int((offset[0] + image.shape[1]) / region_scale)) + " to " + str(int(offset[0] / region_scale)) +
                       ", mask height: " + str(int((offset[1] + image.shape[0]) / region_scale)) + " to " + str(int(offset[1] / region_scale)) + ", Image:" + str(image.shape) + ", " + str(image.dtype) + ", Mask:" + str(maskSection.shape) + ", "  + str(maskSection.dtype))

    resize_factor = (1.0, 1.0)

    try:
        t = time.time()
        image, resize_factor = model.predict_and_interpret(image, image_scale, equalize)
        # print("predicted time:", time.time() - t)
    except Exception:
        print_error("Failed to make predictions on Image: " + str(img_data[0]))

    try:
        t = time.time()
        #use openCVs experimental fast edge based line detector (I find it works better than their regular one)
        lines = extract_lines(image)

        if lines is not None:
            lines = lines.tolist()

            lines = [l[0] for l in lines]

            #converting pixels to coordinate offset pixels so the lines are relative to lines for other images
            lines = [[l[0]/resize_factor[0] + offset[0], l[1]/resize_factor[1] + offset[1], l[2]/resize_factor[0] + offset[0], l[3]/resize_factor[1] + offset[1]] for l in lines]

            all_lines.extend(lines)
            # print("lines time:", time.time() - t, "total lines:", len(all_lines))
    except Exception:
        print_error("Failed to detect lines: " + str(img_data[0]))

print("Joining lines")

try:
    #once all lines have been found, they need to be joined
    #see C++ fastLib for more details
    all_lines = join_lines(all_lines, top_left, bottom_right, image_scale, parameters[2:])
except Exception:
    print_error("Failed to join lines: make sure the entered settings are correct, and that FastLib is in the python folder in the Tree Tagger installation folder")

#exporting lines to shape file
print("Exporting lines to shape files")
project_pathExtension = "/TreeTagger/Results_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
try:
    os.mkdir(project_path + project_pathExtension)
    file_path = project_path + project_pathExtension + '/Lines_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_line_shapefile(all_lines, image_scale, top_left, file_path)
except Exception:
    print_error("couldn't create line shapefile, make sure TreeTagger folder exists in your ArcGIS project directory")

print("Detecting Tree Directions")
# converting the now joined lines from pixel to coordinates
# also keeping track of their current position in the list
coord_lines = pixels_to_coords(all_lines, image_scale, top_left)

for i in range(len(coord_lines)):
    coord_lines[i].append(i)

#PIL has a safety feature for image bombs, we actually have massive images, so this is turned off
Image.MAX_IMAGE_PIXELS = None

#creating a list of all images being process or bordering an image being processed
combined_data = images_data + border_data

#list of all images sizes to that an image doesn't have to be partially loaded each time it is checked
image_sizes = []

for data in combined_data:
    im = Image.open(data[0])
    image_sizes.append(im.size)

model = treetag.TreeDirectionModel(os.path.dirname(os.path.realpath(__file__)) + "\\weights59.h5")

#all validly predicted lines
predicted_lines = []
#inconclusive lines
inc_lines = []

#for each image being processed, check all other images to see if there exists images which exactly border it in a grid
#Unfortunately, the best way I could accurately do this is to manually check if each image fit exactly to the
#left, right, top, bottom or 4 corners of the current image.
for img_data in tqdm(images_data, file=sys.stdout):
    image = cv2.imread(img_data[0], cv2.IMREAD_COLOR)
    if image.shape[0] < 2048 or image.shape[1] < 2048:
        continue
    original_size = image.shape

    extended_region = find_surrounding_images(image, img_data, image_scale, combined_data, image_sizes, line_extension_size)

    #offset for maximum distance line could be extended off the current image
    resize_offset = (int(math.ceil(line_extension_size / image_scale[0])*2), int(math.ceil(line_extension_size / image_scale[1]))*2)
    #image extended on all sizes by resize offset
    extended_image = np.zeros((image.shape[0] + 2*resize_offset[1], image.shape[1] + 2*resize_offset[0], 3), np.uint8)
    #moving image to center of extended image
    extended_image[resize_offset[1]:resize_offset[1]+image.shape[0], resize_offset[0]:resize_offset[0]+image.shape[1]] = image

    image = crop_extended_region(image, extended_region, resize_offset)

    if equalize:
        image = equalize_image(image)

    #lines contained within current image
    image_lines, coord_lines = find_lines_in_image(img_data, original_size, image_scale, coord_lines)

    #convert coordinates to current image pixels
    #image_lines = coords_to_pixels(image_lines, image_scale, img_data[1:3])
    image_lines = [[int((l[0] - img_data[1])/image_scale[0]), int((img_data[2] - l[1])/image_scale[1]), int((l[2] - img_data[1])/image_scale[0]), int((img_data[2] - l[3])/image_scale[1]), l[4]] for l in image_lines]

    #split lines into batches of 32 to be feed through the CNN in parallel
    image_line_batch_size = 256

    image_line_batches = split_lines_into_batches(image_lines, image_line_batch_size)

    #for each batch, crop out the sections around each tree, rotate them to be horizontal and then feed the batch
    #of images into the CNN and interpret the results
    for b in tqdm(range(len(image_line_batches)), file=sys.stdout, leave=False):
        batch = image_line_batches[b]
        batch_images = batch_line_boxes(image, batch, extended_region, resize_offset)

        #if all lines in the batch were skipped
        if len(batch_images) == 0:
            continue

        #feed batch into CNN to get prediction
        predicted, inc = model.predict_and_interpret(batch, batch_images, all_lines, batch_size=image_line_batch_size)
        predicted_lines.extend(predicted)
        inc_lines.extend(inc)

print("Exporting direction lines to shape files")
try:
    file_path = project_path + project_pathExtension + '/Vectors_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_vector_shapefile(predicted_lines, image_scale, top_left, file_path)
except Exception:
    print_error("couldn't create refined lines shapefile, make sure TreeTagger folder exists in your ArcGIS project directory")


print("getting average directions")
grid_vectors = build_vectors(direction_grid_parameters, direction_interpolation, top_left, bottom_right, predicted_lines, image_scale)

#since we now have the directions for any non-false positive / inclusive lines, we can extend small lines in the
#direction they're pointing and the attempt to join the lines once again
print("Augmenting lines")
all_lines = refine_lines(predicted_lines, image_scale, top_left, bottom_right, parameters[2:])

print("Exporting refined lines to shape files")
try:
    file_path = project_path + project_pathExtension + '/Refined_Lines_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_line_shapefile(all_lines, image_scale, top_left, file_path)
except Exception:
    print_error("couldn't create refined lines shapefile, make sure TreeTagger folder exists in your ArcGIS project directory")

print("Exporting directions to shape files")
try:
    file_path = project_path + project_pathExtension + '/Directions_'
    export_direction_shapefile(grid_vectors, file_path, direction_grid_parameters, direction_interpolation)
except Exception:
    print_error("couldn't create direction shapefile, make sure TreeTagger folder exists in your ArcGIS project directory")

print("Exporting directions to csv")
try:
    file_path = project_path + project_pathExtension + '\\Directions_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
    export_direction_csv(grid_vectors, file_path)
except Exception:
    print_error("couldn't create direction csv, make sure TreeTagger folder exists in your ArcGIS project directory")

# detecting high density regions by using the dbscan clustering algorithm to group high density areas together
# and then fitting a polygon to enclose the high density clusters
try:
    print("Calculating high density regions")
    high_density_regions = find_high_density_areas(all_lines, parameters, top_left, image_scale)

    print("exporting regions to shapefile")
    file_path = project_path + project_pathExtension + '/Regions_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_polygon_shapefile(high_density_regions, file_path)

except Exception:
    print_error("Error detecting high density regions, make sure the entered settings are correct")

#program has finished, print total run time in seconds
print("time: ", timeit.default_timer() - t)




