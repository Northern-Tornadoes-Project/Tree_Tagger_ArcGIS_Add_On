'''
    Main code for automated tree tagging and analysis software
    Author: Daniel Butt, NTP 2022
    Email: dbutt7@uwo.ca
    Date of last revision: Aug 9, 2022
'''

import math
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
import tensorflow as tf
from keras_segmentation.models.unet import resnet50_unet
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import FastLib
import timeit
import shapefile
import os
from datetime import datetime
import traceback
from PIL import Image
import csv

#function for loading the FCN model for extracting the tree mask
def load_FCN_model():
    model = resnet50_unet(n_classes=2, input_height=256, input_width=256)
    model.load_weights(os.path.dirname(os.path.realpath(__file__)) + "\\weights30.h5")

    return model

#function for loading the CNN model for predicting tree directions
def load_CNN_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.applications.resnet.ResNet50(include_top=False, input_shape=(128, 256, 3), classes=3, pooling=max, weights=None))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(64, activation='relu'))

    model.add(tf.keras.layers.Dense(32, activation='relu'))

    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.load_weights(os.path.dirname(os.path.realpath(__file__)) + "\\weights9_.h5")

    return model


#Rotates an image (angle in degrees) and expands image to avoid cropping
def rotate_image(mat, angle):

    height, width = mat.shape[:2]
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origin) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


# function for determining if a piece of data is a float
def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False


# function for printing an error and then exiting
def printError(msg):
    print("TreeTagger Python Error:\n", msg, traceback.format_exc(), file=sys.stderr)
    exit(1)

# timer to measure total runtime
t = timeit.default_timer()

#checking if tensorflow was built with cuda support and the user has a compatible gpu
if tf.test.is_built_with_cuda():
    print("CUDA ENABLED")
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) != 0:
        print("Running on: ", devices[0])

#list of arguments passed from arcgis add in
argv = []
# path to arcgis project folder
projectPath = ""

print("Loading arguments")
try:
    #geting project path
    argv = sys.argv[1:]
    projectPath = argv[0]
except IndexError:
    printError("Couldn't load Tree Tagger settings, make sure the entered settings are correct")

try:
    #reading arguments file created by arcgis add in
    argsFile = open(projectPath + "\\TreeTagger\\args.txt")
    #arguments are separated using | since it can't be put in a file name
    argv = argsFile.readline().split("|")
except FileNotFoundError:
    printError("Couldn't load Tree Tagger arguments file, make sure the entered settings are correct")

#parameters for joining lines and finding high density areas
parameters = []
#image file paths and coordinates
imagesData = []
#image file paths and coordinates for images bordering the images being processed
borderData = []
#the points of the polygon region selected, if a region was selected
polygonPoints = []
#image scale (usually 5cm imagery so (0.05, 0.05))
imageScale = (0, 0)
#coordinate of bottom right corner
bottomRight = (0, 0)
#coordinate of top left corner
topLeft = (0, 0)
#whether a polygon region was selected
isPolygon = False
#grid sizes for averaging direction vectors
directionGridSizes = [20, 50, 100]
#size to extend line for cropping section around a tree before assessing the trees direction
lineExtensionSize = 12.8

# loading and organizing all arguments passed from arcgis add in
# see add in code from the order of arguments
try:

    directionGridSizes = [int(x) for x in argv[:4]]

    argv = argv[4:]

    parameters = [float(x) for x in argv[:7]]

    argv = argv[7:]

    imageScale = (round(float(argv[0]), 3), round(float(argv[1]), 3))

    topLeft = (float(argv[2]), float(argv[3]))

    bottomRight = (float(argv[4]), float(argv[5]))

    isPolygon = bool(int(argv[6]))

    argv = argv[7:]

    x = 0
    if isPolygon:
        while is_float(argv[x]):
            polygonPoints.append((float(argv[x]), float(argv[x+1])))
            x += 2

    argv = argv[x:]
    imgArgs = []
    borderArgs = []

    i = 0
    while i < len(argv) and "?" not in argv[i]:
        imgArgs.append(argv[i])
        i += 1

    i += 1

    while(i < len(argv)):
        borderArgs.append(argv[i])
        i += 1

    for i in range(0, len(imgArgs), 3):
        imagesData.append((imgArgs[i], float(imgArgs[i + 1]), float(imgArgs[i + 2])))

    for i in range(0, len(borderArgs), 3):
        borderData.append((borderArgs[i], float(borderArgs[i + 1]), float(borderArgs[i + 2])))

except Exception:
    printError("Couldn't parse Tree Tagger arguments file, make sure the entered settings are correct ")

#if a polygon region was selected, the size of the polygon is scaled down to save space when creating a mask
#to bitwise AND with each image being processed. The size of the scaled down mask depends on the amount of pixels
#which would encompass the bounding box the polygon (topLeft -> bottomRight)
#The polygon is then resized to fit into an image with a fixed number of pixels
#
#total pixels in image = w * h, fixed pixel sized image = (original width/scale factor)*(original height/scale factor)
#scale factor = sqrt((original width * original height) / fixed pixel size)

regionWidth = (bottomRight[0] - topLeft[0]) / imageScale[0]
regionHeight = (topLeft[1] - bottomRight[1]) / imageScale[1]
regionScale = math.sqrt((regionWidth * regionHeight) / 1000000)

#make total pixels larger if bounding box is massive
if regionHeight >= 500000 or regionWidth >= 500000:
    regionScale = math.sqrt((regionWidth * regionHeight) / 1000000000)
elif regionHeight >= 50000 or regionWidth >= 50000:
    regionScale = math.sqrt((regionWidth * regionHeight) / 100000000)

# mask for polygon
mask = np.zeros([max(int(math.ceil(regionHeight / regionScale)), 1), max(int(math.ceil((regionWidth / regionScale))), 1)], np.uint8)

if regionWidth == 0 or regionHeight == 0 or regionScale == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
    printError("invalid region selection, Region Width: " + str(regionWidth) + ", Region Height: " + str(regionHeight) +
               ", Region Scale: " + str(regionScale) + "Make sure you selected an image/polygon region which contains an image, also check that the coordinate systems match")

if isPolygon:
    print("Cropping out polygon region")
    #convert polygon points from coords to pixels and then resize the polygon to fit fixed scale factor
    polygonPoints = [[int(((p[0] - topLeft[0]) / imageScale[0]) / regionScale), int(((topLeft[1] - p[1]) / imageScale[1]) / regionScale)] for p in polygonPoints]
    try:
        #draw filled polygon on mask image so that it can be bitwise ANDed effectively
        cv2.drawContours(mask, [np.array(polygonPoints, np.int)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    except Exception:
        printError("Couldn't create mask for selected polygon, check that the coordinate systems match")


print("Loading machine learning model")
try:
    model = load_FCN_model()
except Exception:
    printError("Couldn't load model, check Tree Tagger installation folder for weights file / Keras_Segmentation package")

print("Calculating predictions and detecting lines")
#all the lines detected between all images being processed
allLines = []

for imgData in tqdm(imagesData, file=sys.stdout):
    #offset to keep track of where lines are relative to the coordinates of the image being processed
    offset = ((imgData[1] - topLeft[0]) / imageScale[0], (topLeft[1] - imgData[2]) / imageScale[1])
    image = 0

    #loading the current image being processed and make sure its size and color bands (RGB and not gray scale, etc.)
    try:
        image = cv2.imread(imgData[0], cv2.IMREAD_COLOR)
    except Exception:
        printError("Couldn't load image: " + str(imgData[0]))

    if image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] != 3:
        printError("invaild image: image width: " + str(image.shape[1]) + ", image height: " + str(image.shape[0]) + ", image color bands (rgb should be 3): " + str(image.shape[2]))

    #if a polygon region was selected, crop out and resize the fixed sized polygon mask so that it's the same size as
    #the image being processed. Then bitwise AND the image and mask to black out any region not contained in the polygon
    if isPolygon:
        try:
            xOffset = 0
            yOffset = 0

            if int(offset[1] / regionScale) == int((offset[1] + image.shape[0]) / regionScale):
                yOffset = 1
            if int(offset[0] / regionScale) == int((offset[0] + image.shape[1]) / regionScale):
                xOffset = 1

            maskSection = mask[int(offset[1] / regionScale):int((offset[1] + image.shape[0]) / regionScale) + yOffset, int(offset[0] / regionScale):int((offset[0] + image.shape[1]) / regionScale) + xOffset]
            maskSection = cv2.resize(maskSection, (image.shape[1], image.shape[0]), cv2.INTER_CUBIC)
            image = cv2.bitwise_and(image, image, mask=maskSection)
        except Exception:
            printError("Couldn't extract polygon mask, Make sure you selected an image/polygon region which contains an"
                       + " image, also check that the coordinate systems match, mask width: " +
                       str(int((offset[0] + image.shape[1]) / regionScale)) + " to " + str(int(offset[0] / regionScale)) +
                       ", mask height: " + str(int((offset[1] + image.shape[0]) / regionScale)) + " to " + str(int(offset[1] / regionScale)) + ", Image:" + str(image.shape) + ", " + str(image.dtype) + ", Mask:" + str(maskSection.shape) + ", "  + str(maskSection.dtype))

    resize_factor = (1.0, 1.0)

    try:
        # since the FCN model was trained on 5cm imagery, it expects a tree to be a certain size in terms of pixels and look
        # certain ways, which wouldn't be the case with higher quality imagery
        # as such the image should be resized to match 5cm imagery better
        if imageScale[0] != 0.05 or imageScale[1] != 0.05:
            resize_factor = (imageScale[0] / 0.05, imageScale[1] / 0.05)
            image = cv2.resize(image, (image.shape[1] * resize_factor[0], image.shape[0] * resize_factor[1]), cv2.INTER_CUBIC)

        resized_shape = (image.shape[0], image.shape[1])

        #image is again resized so that it is a multiple of 256 as required by the FCN
        y = int(max(image.shape[0] / 256.0, 1))
        x = int(max(image.shape[1] / 256.0, 1))
        image = cv2.resize(image, (x*256, y*256), cv2.INTER_AREA)

        #split the image into 256x256 sections to feed into the FCN model
        #image pixels are also converted to floats and normalized
        sections = []
        for i in range(y):
            for j in range(x):
                section = image[i * 256:i * 256 + 256, j * 256:j * 256 + 256]
                section = section.astype(np.float32)
                section = section/255.0
                sections.append(section)

        #needs to be numpy array
        sections = np.asarray(sections)

        #get the FCN model to make predictions for each 256x256 section
        predictions = model.predict(sections, batch_size=16)

        #convert predictions into appropriate pixels and stitch image back together
        for i in range(y):
            for j in range(x):
                prediction = np.argmax(predictions[i*x + j], axis=1)
                prediction = np.reshape(prediction, (128, 128, 1))
                prediction = prediction.astype(np.uint8)
                prediction *= 255
                prediction = cv2.resize(prediction, (256, 256), cv2.INTER_CUBIC)
                prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

                image[i * 256:i * 256 + 256, j * 256:j * 256 + 256] = prediction

        #resize image back to original size (after being resized to 5cm imagery)
        image = cv2.resize(image, (resized_shape[1], resized_shape[0]), cv2.INTER_CUBIC)

        #convert to gray scale as color is no longer required
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        printError("Failed to make predictions on Image: " + str(imgData[0]))

    try:
        #use openCVs experimental fast edge based line detector (I find it works better than their regular one)
        lsd = cv2.ximgproc.createFastLineDetector(do_merge=False)

        lines = lsd.detect(image)

        if lines is not None:
            lines = lines.tolist()

            lines = [l[0] for l in lines]

            #converting pixels to coordinate offset pixels so the lines are relative to lines for other images
            lines = [[l[0]/resize_factor[0] + offset[0], l[1]/resize_factor[1] + offset[1], l[2]/resize_factor[0] + offset[0], l[3]/resize_factor[1] + offset[1]] for l in lines]

            allLines.extend(lines)
    except Exception:
        printError("Failed to detect lines: " + str(imgData[0]))

print("Joining lines")

try:
    #once all lines have been found, they need to be joined
    #see C++ fastLib for more details
    x = int(math.ceil((bottomRight[0] - topLeft[0]) / imageScale[0] / 256.0))
    y = int(math.ceil((topLeft[1] - bottomRight[1]) / imageScale[1] / 256.0))

    allLines = FastLib.joinLines(allLines, x, y, parameters[2], parameters[3], parameters[4], parameters[5], parameters[6])
except Exception:
    printError("Failed to join lines: make sure the entered settings are correct, and that FastLib is in the python folder in the Tree Tagger installation folder")

#exporting lines to shape file
print("Exporting lines to shape files")
try:
    #converting pixels to coords
    coordLines = [[topLeft[0] + l[0] * imageScale[0], topLeft[1] - l[1] * imageScale[1], topLeft[0] + l[2] * imageScale[0], topLeft[1] - l[3] * imageScale[1]] for l in allLines]

    shpw = shapefile.Writer(projectPath + '/TreeTagger/Lines_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), shapeType=3)
    shpw.field('ID', 'N')

    x = 0
    for l in coordLines:
        shpw.line([[[l[0], l[1]], [l[2], l[3]]]])
        #shape file must have some kind of record (label) for each shape
        shpw.record(x)
        x += 1

    shpw.close()
except Exception:
    printError("couldn't create line shapefile, make sure TreeTagger folder exists in your ArcGIS project directory")

print("Detecting Tree Directions")
# converting the now joined lines from pixel to coordinates
# also keeping track of their current position in the list
coordLines = [[topLeft[0] + l[0] * imageScale[0], topLeft[1] - l[1] * imageScale[1], topLeft[0] + l[2] * imageScale[0], topLeft[1] - l[3] * imageScale[1]] for l in allLines]
for i in range(len(coordLines)):
    coordLines[i].append(i)

#PIL has a safety feature for image bombs, we actually have massive images, so this is turned off
Image.MAX_IMAGE_PIXELS = None

#creating a list of all images being process or bordering an image being processed
combinedData = imagesData + borderData

#The tree direction CNN needs to analysis a box around each tree. Since that box could intersect multiple images, we
#have to know if their exists images which surround the current image being processed (3x3 grid)
#since some images for events are not in a perfect grid (skewed / not all the same size, etc.) this is a bit of a pain.
#Also, to save memory the images are loaded one at a time and the bordering section is cropped out
extendedRegion = [["None"]*3,
                  ["None"]*3,
                  ["None"]*3]

#list of all images sizes to that an image doesn't have to be partially loaded each time it is checked
imageSizes = []

for data in combinedData:
    im = Image.open(data[0])
    imageSizes.append(im.size)

model = load_CNN_model()

#all validly predicted lines
predicted_lines = []
#inconclusive lines
inc_lines = []

#for each image being processed, check all other images to see if there exists images which exactly border it in a grid
#Unfortunately, the best way I could accurately do this is to manually check if each image fit exactly to the
#left, right, top, bottom or 4 corners of the current image.
for imgData in tqdm(imagesData, file=sys.stdout):
    image = cv2.imread(imgData[0], cv2.IMREAD_COLOR)
    originalSize = image.shape

    for i in range(len(combinedData)):
        data = combinedData[i]
        width = imageSizes[i][0] * imageScale[0]
        height = imageSizes[i][1] * imageScale[1]


        # top left corner
        if extendedRegion[0][0] == "None" and width >= lineExtensionSize and height >= lineExtensionSize and round(data[1] + width) == round(imgData[1]) and round(data[2] - height) == round(imgData[2]):
            extendedRegion[0][0] = data[0]

        # top
        elif extendedRegion[0][1] == "None" and height >= lineExtensionSize and round(data[1]) == round(imgData[1]) and round(data[2] - height) == round(imgData[2]):
            extendedRegion[0][1] = data[0]

        # top right corner
        elif extendedRegion[0][2] == "None" and width >= lineExtensionSize and height >= lineExtensionSize and round(data[1]) == round(imgData[1] + image.shape[1]*imageScale[0]) and round(data[2] - height) == round(imgData[2]):
            extendedRegion[0][2] = data[0]

        # left
        elif extendedRegion[1][0] == "None" and width >= lineExtensionSize and round(data[1] + width) == round(imgData[1]) and round(data[2]) == round(imgData[2]):
            extendedRegion[1][0] = data[0]

        #right
        elif extendedRegion[1][2] == "None" and width >= lineExtensionSize and round(data[1]) == round(imgData[1] + image.shape[1]*imageScale[0]) and round(data[2]) == round(imgData[2]):
            extendedRegion[1][2] = data[0]

        #bottom left corner
        elif extendedRegion[2][0] == "None" and width >= lineExtensionSize and height >= lineExtensionSize and round(data[1] + width) == round(imgData[1]) and round(data[2]) == round(imgData[2] - image.shape[0]*imageScale[1]):
            extendedRegion[2][0] = data[0]

        #bottom
        elif extendedRegion[2][1] == "None" and height >= lineExtensionSize and round(data[1]) == round(imgData[1]) and round(data[2]) == round(imgData[2] - image.shape[0]*imageScale[1]):
            extendedRegion[2][1] = data[0]

        #bottom right corner
        elif extendedRegion[2][2] == "None" and width >= lineExtensionSize and height >= lineExtensionSize and round(data[1]) == round(imgData[1] + image.shape[1]*imageScale[0]) and round(data[2]) == round(imgData[2] - image.shape[0]*imageScale[1]):
            extendedRegion[2][2] = data[0]

    #offset for maximum distance line could be extended off the current image
    resizeOffset = (int(math.ceil(lineExtensionSize / imageScale[0])*2), int(math.ceil(lineExtensionSize / imageScale[1]))*2)
    #image extended on all sizes by resize offset
    extendedImage = np.zeros((image.shape[0] + 2*resizeOffset[1], image.shape[1] + 2*resizeOffset[0], 3), np.uint8)
    #moving image to center of extended image
    extendedImage[resizeOffset[1]:resizeOffset[1]+image.shape[0], resizeOffset[0]:resizeOffset[0]+image.shape[1]] = image

    #top left corner
    if extendedRegion[0][0] != "None":
        image = cv2.imread(extendedRegion[0][0], cv2.IMREAD_COLOR)
        extendedImage[0:resizeOffset[1], 0:resizeOffset[0]] = image[image.shape[0]-resizeOffset[1]:image.shape[0], image.shape[1]-resizeOffset[0]:image.shape[1]]

    #top
    if extendedRegion[0][1] != "None":
        image = cv2.imread(extendedRegion[0][1], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (originalSize[1], image.shape[0]), cv2.INTER_CUBIC)
        extendedImage[0:resizeOffset[1], resizeOffset[0]:resizeOffset[0]+image.shape[1]] = image[image.shape[0] - resizeOffset[1]:image.shape[0], 0:image.shape[1]]

    #top right corner
    if extendedRegion[0][2] != "None":
        image = cv2.imread(extendedRegion[0][2], cv2.IMREAD_COLOR)
        extendedImage[0:resizeOffset[1], extendedImage.shape[1]-resizeOffset[0]:extendedImage.shape[1]] = image[image.shape[0]-resizeOffset[1]:image.shape[0], 0:resizeOffset[0]]

    #left
    if extendedRegion[1][0] != "None":
        image = cv2.imread(extendedRegion[1][0], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (image.shape[1], originalSize[0]), cv2.INTER_CUBIC)
        extendedImage[resizeOffset[1]:resizeOffset[1]+image.shape[0], 0:resizeOffset[0]] = image[0:image.shape[0], image.shape[1]-resizeOffset[0]:image.shape[1]]

    #right
    if extendedRegion[1][2] != "None":
        image = cv2.imread(extendedRegion[1][2], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (image.shape[1], originalSize[0]), cv2.INTER_CUBIC)
        extendedImage[resizeOffset[1]:resizeOffset[1]+image.shape[0], extendedImage.shape[1]-resizeOffset[0]:extendedImage.shape[1]] = image[0:image.shape[0], 0:resizeOffset[0]]

    #bottom left corner
    if extendedRegion[2][0] != "None":
        image = cv2.imread(extendedRegion[2][0], cv2.IMREAD_COLOR)
        extendedImage[extendedImage.shape[0]-resizeOffset[1]:extendedImage.shape[0], 0:resizeOffset[0]] = image[0:resizeOffset[1], image.shape[1]-resizeOffset[0]:image.shape[1]]

    #bottom
    if extendedRegion[2][1] != "None":
        image = cv2.imread(extendedRegion[2][1], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (originalSize[1], image.shape[0]), cv2.INTER_CUBIC)
        extendedImage[extendedImage.shape[0]-resizeOffset[1]:extendedImage.shape[0], resizeOffset[0]:resizeOffset[0] + image.shape[1]] = image[0:resizeOffset[1], 0:image.shape[1]]

    #bottom right corner
    if extendedRegion[2][2] != "None":
        image = cv2.imread(extendedRegion[2][2], cv2.IMREAD_COLOR)
        extendedImage[extendedImage.shape[0]-resizeOffset[1]:extendedImage.shape[0], extendedImage.shape[1]-resizeOffset[0]:extendedImage.shape[1]] = image[0:resizeOffset[1], 0:resizeOffset[0]]

    image = extendedImage

    #lines contained within current image
    imageLines = []

    #if the line is in the image add it to image lines and remove it from coordlines as to not check it twice
    for line in coordLines[:]:
        if (imgData[1] <= line[0] <= (imgData[1] + originalSize[1] * imageScale[0]) and (imgData[2] - originalSize[0] * imageScale[1]) <= line[1] <= imgData[2]) or (imgData[1] <= line[2] <= (imgData[1] + originalSize[1] * imageScale[0]) and (imgData[2] - originalSize[0] * imageScale[1]) <= line[3] <= imgData[2]):
            imageLines.append(line)
            coordLines.remove(line)

    #convert coordinates to current image pixels
    imageLines = [[int((l[0] - imgData[1])/imageScale[0]), int((imgData[2] - l[1])/imageScale[1]), int((l[2] - imgData[1])/imageScale[0]), int((imgData[2] - l[3])/imageScale[1]), l[4]] for l in imageLines]
    imageLineBatches = []

    #split lines into batches of 32 to be feed through the CNN in parallel
    i = 0
    while i + 32 < len(imageLines):
        imageLineBatches.append(imageLines[i:i+32])
        i += 32
    if i + 1 < len(imageLines):
        imageLineBatches.append(imageLines[i:i+(len(imageLines)-i)])

    #for each batch, crop out the sections around each tree, rotate them to be horizontal and then feed the batch
    #of images into the CNN and interpret the results
    for b in tqdm(range(len(imageLineBatches)), file=sys.stdout, leave=False):
        batch = imageLineBatches[b]
        batchImages = []
        #for each line in batch
        for i in range(len(batch)):
            line = batch[i]
            #offset the line based on extended border of image
            line = [line[0] + resizeOffset[0], line[1] + resizeOffset[1], line[2] + resizeOffset[0], line[3] + resizeOffset[1], line[4]]

            #if line is perfectly horizontal / vertical shift it by one pixel (avoids multiplying / dividing by 0)
            if line[0] == line[2]:
                line[2] += 1

            if line[1] == line[3]:
                line[3] += 1

            #mid-point of the line
            mid_point = [round((line[0] + line[2]) / 2.0), round((line[1] + line[3]) / 2.0)]
            #angle of the line
            angle = math.atan((line[3] - line[1]) / (line[2] - line[0] + 0.00001)) * (180.0 / 3.14159265358979)
            #slope of the line
            slope = float(line[3] - line[1]) / float(line[2] - line[0])
            #negative reciprocal slope (perpendicular slope)
            nr_slope = -1.0 / slope

            #scale of x extension to extend line
            scale = resizeOffset[0] / math.sqrt(1 + slope ** 2)

            extended_line = [int(mid_point[0] - scale), int(mid_point[1] - slope * scale),
                             int(mid_point[0] + scale), int(mid_point[1] + slope * scale)]

            #scale of x in the perpendicular direction to create box around the tree
            scale = (resizeOffset[0] / 4.0) / math.sqrt(1 + nr_slope ** 2)

            skip = False

            #points of the box to crop out section around tree for rotation
            box_points = np.array([[int(extended_line[0] - scale), int(extended_line[1] - nr_slope * scale)],
                                   [int(extended_line[0] + scale), int(extended_line[1] + nr_slope * scale)],
                                   [int(extended_line[2] + scale), int(extended_line[3] + nr_slope * scale)],
                                   [int(extended_line[2] - scale), int(extended_line[3] - nr_slope * scale)]], np.int32)

            #make sure the box fits within the extended image
            for p in box_points:
                #in main image
                if resizeOffset[0] < p[0] < image.shape[1] - resizeOffset[0] and resizeOffset[1] < p[1] < image.shape[0] - resizeOffset[1]:
                    continue
                #not in extended image
                if p[0] < 0 or p[0] >= image.shape[1] or p[1] < 0 or p[1] >= image.shape[0]:
                    skip = True
                    break
                #left side of extended
                elif p[0] < resizeOffset[0] and extendedRegion[1][0] == "None":
                    skip = True
                    break
                #right side of extended
                elif p[0] > image.shape[1] - resizeOffset[0] and extendedRegion[1][2] == "None":
                    skip = True
                    break
                #top of extended
                elif p[1] < resizeOffset[1] and extendedRegion[0][1] == "None":
                    skip = True
                    break
                #bottom of extended
                elif p[1] > image.shape[0] - resizeOffset[1] and extendedRegion[2][1] == "None":
                    skip = True
                    break
                #top left corner
                elif p[0] < resizeOffset[0] and p[1] < resizeOffset[1] and extendedRegion[0][0] == "None":
                    skip = True
                    break
                #top right corner
                elif p[0] > image.shape[1] - resizeOffset[0] and p[1] < resizeOffset[1] and extendedRegion[0][2] == "None":
                    skip = True
                    break
                #bottom right corner
                elif p[0] > image.shape[1] - resizeOffset[0] and p[1] > image.shape[0] - resizeOffset[1] and extendedRegion[2][2] == "None":
                    skip = True
                    break
                #bottom left corner
                elif p[0] < resizeOffset[0] and p[1] > image.shape[0] - resizeOffset[1] and extendedRegion[2][0] == "None":
                    skip = True

            #if the box doesn't fit within the extended image, mark the line for be skipped later
            if skip:
                batch[i][4] = -1
                continue

            #sort bounding box points so that we can crop out a non-rotated box around the box which surround the tree
            bounding_box = [[image.shape[1], image.shape[0]], [0, 0]]

            for p in box_points:
                if p[0] < bounding_box[0][0]:
                    bounding_box[0][0] = p[0]
                if p[0] > bounding_box[1][0]:
                    bounding_box[1][0] = p[0]
                if p[1] < bounding_box[0][1]:
                    bounding_box[0][1] = p[1]
                if p[1] > bounding_box[1][1]:
                    bounding_box[1][1] = p[1]

            #offseting box points so that the top left corner of the box is at the origin
            box_points_offset = np.array([[p[0] - bounding_box[0][0], p[1] - bounding_box[0][1]] for p in box_points],
                                         np.int)

            #cropping out the box surrounding the tree box
            box_image = np.copy(image[bounding_box[0][1]:bounding_box[1][1], bounding_box[0][0]:bounding_box[1][0]])

            #drawing the found line over the tree
            cv2.line(box_image, [line[0] - bounding_box[0][0], line[1] - bounding_box[0][1]],
                     [line[2] - bounding_box[0][0], line[3] - bounding_box[0][1]], [0, 0, 255], 1, cv2.LINE_AA)

            #mask to extract the tree box
            mask = np.zeros((bounding_box[1][1] - bounding_box[0][1], bounding_box[1][0] - bounding_box[0][0]), np.uint8)

            #drawing box to build mask
            cv2.drawContours(mask, [box_points_offset], -1, 255, -1)

            #bitwise AND to extract mask
            cv2.bitwise_and(box_image, box_image, mask=mask)

            #rotating image to make it horizontal
            rotated_image = rotate_image(box_image, angle)

            #cropping out the tree box and stretching it
            mid_point = [int(rotated_image.shape[1] / 2), int(rotated_image.shape[0] / 2)]

            box_image = rotated_image[mid_point[1] - 64:mid_point[1] + 64, mid_point[0] - 256:mid_point[0] + 256]

            box_image = cv2.resize(box_image, (256, 128), interpolation=cv2.INTER_NEAREST)

            #making the tree box image a numpy float32 array
            pImage = np.array(box_image, np.float32)
            #normalizing the image
            pImage /= 255.0
            #adding it to a batch to be processed in parallel
            batchImages.append(pImage)

        #if all lines in the batch were skipped
        if len(batchImages) == 0:
            continue

        #feed batch into CNN to get prediction
        predictions = model.predict(np.asarray(batchImages))

        #interpret results of predictions
        # k for counter... to keep track of lines skipped
        k = 0
        #for each line in batch
        for i in range(len(batch)):
            line = batch[i]
            #if line was skipped
            if line[4] < 0:
                continue

            #get the original line
            trueLine = allLines[line[4]]

            #get the prediction for the line's direction
            j = np.argmax(predictions[k], axis=0)

            #left
            if j == 0:
                predicted_lines.append([trueLine[2], trueLine[3], trueLine[0], trueLine[1], np.amax(predictions[k], axis=0)])

            #right
            elif j == 2:
                predicted_lines.append([trueLine[0], trueLine[1], trueLine[2], trueLine[3], np.amax(predictions[k], axis=0)])

            #inconclusive
            else:
                inc_lines.append(trueLine)

            #increase counter since the line wasn't skipped
            k += 1

print("getting average directions")
grid_directions = []
grid_size_min_lines = [1, 3, 10, 50]

#for each grid direction size (small, med, large)
for i in range(len(directionGridSizes)):
    size = directionGridSizes[i]

    #get the weighted average grid directions
    x = int(math.ceil((bottomRight[0] - topLeft[0]) / size))
    y = int(math.ceil((topLeft[1] - bottomRight[1]) / size))

    #see Fast lib for more info
    grid_directions.append(FastLib.averageDirections(predicted_lines, (size/imageScale[0]), x, y, i + 5, grid_size_min_lines[i]))

grid_vectors = []

#using the average grid directions, build vectors to be displayed by arcgis
print("Building vectors")
for k in range(len(directionGridSizes)):
    size = directionGridSizes[k]

    x = int(math.ceil((bottomRight[0] - topLeft[0]) / size))
    y = int(math.ceil((topLeft[1] - bottomRight[1]) / size))

    grid_vectors.append([])

    for i in range(y):
        for j in range(x):
            gridVec = grid_directions[k][i][j]

            if gridVec[2] == 0:
               continue

            grid_vectors[k].append([topLeft[0] + j*size + size/2,
                                    topLeft[1] - i*size - size/2,
                                    topLeft[0] + j*size + size/2 + gridVec[0] * size/2,
                                    topLeft[1] - i*size - size/2 - gridVec[1] * size/2,
                                    gridVec[2]])

#since we now have the directions for any non-false positive / inclusive lines, we can extend small lines in the
#direction they're pointing and the attempt to join the lines once again
minSize = 5.0 #5 meters
print("Augmenting lines")
for line in predicted_lines:
    if math.sqrt((line[3] - line[1]) ** 2 + (line[2] - line[0]) ** 2) < (minSize/imageScale[0]):
        if line[2] == line[0]:
            line[2] += 1
        slope = float(line[3] - line[1]) / float(line[2] - line[0])
        scale = (minSize/imageScale[0]) / math.sqrt(1 + slope ** 2)
        line[2] = int(line[0] + math.copysign(scale, line[2] - line[0]))
        line[3] = int(line[1] + math.copysign(scale * slope, line[3] - line[1]))

x = int(math.ceil((bottomRight[0] - topLeft[0]) / imageScale[0] / 256.0))
y = int(math.ceil((topLeft[1] - bottomRight[1]) / imageScale[1] / 256.0))

#joining lines again
allLines = [line[0:4] for line in predicted_lines]
allLines = FastLib.joinLines(allLines, x, y, float(0.3), float(0.1), float(5), float(5), float(15))
allLines = [[int(l[0]), int(l[1]), int(l[2]), int(l[3])] for l in allLines]

print("Exporting refined lines to shape files")
try:
    #converting pixels to coords
    coordLines = [[topLeft[0] + l[0] * imageScale[0], topLeft[1] - l[1] * imageScale[1], topLeft[0] + l[2] * imageScale[0], topLeft[1] - l[3] * imageScale[1]] for l in allLines]

    shpw = shapefile.Writer(projectPath + '/TreeTagger/Refined_Lines_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), shapeType=3)
    shpw.field('ID', 'N')

    x = 0
    for l in coordLines:
        shpw.line([[[l[0], l[1]], [l[2], l[3]]]])
        #shape file must have some kind of record (label) for each shape
        shpw.record(x)
        x += 1

    shpw.close()
except Exception:
    printError("couldn't create refined lines shapefile, make sure TreeTagger folder exists in your ArcGIS project directory")

print("Exporting directions to shape files")
try:

    shpw = shapefile.Writer(projectPath + '/TreeTagger/Directions_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), shapeType=3)
    shpw.field('Consistency', 'N')
    shpw.field('Scale', 'N')

    for i in range(len(grid_vectors)):
        for l in grid_vectors[i]:
            shpw.line([[[l[0], l[1]], [l[2], l[3]]]])
            shpw.record(Consistency=l[4], Scale=(i+1))

    shpw.close()
except Exception:
    printError("couldn't create direction shapefile, make sure TreeTagger folder exists in your ArcGIS project directory")

print("Exporting directions to csv")
try:
    header = ['x1', 'y1', 'x2', 'y2', 'consistency', 'scale']
    file = open(projectPath + '\\TreeTagger\\Directions_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv', 'w', newline='')
    csv_writer = csv.writer(file)

    csv_writer.writerow(header)

    for i in range(len(grid_vectors)):
        for l in grid_vectors[i]:
            csv_writer.writerow([l[0], l[1], l[2], l[3], l[4], (i+1)])

    file.close()

except Exception:
    printError("couldn't create direction csv, make sure TreeTagger folder exists in your ArcGIS project directory")


# detecting high density regions by using the dbscan clustering algorithm to group high density areas together
# and then fitting a polygon to enclose the high density clusters
try:
    print("Calculating high density regions")
    linePoints = []
    # create a list of all the endpoints of all the lines
    for l in allLines:
        linePoints.append([int(l[0]), int(l[1])])
        linePoints.append([int(l[2]), int(l[3])])

    #create a numpy array for the list
    linePointsnp = np.array(linePoints)

    #run the dbscan clustering algorithm on the set of points (Note 2 points per line)
    dbscan = DBSCAN(eps=parameters[0], min_samples=(parameters[1] * 2)).fit(linePointsnp)

    #get the cluster labels from dbscan
    labels = dbscan.labels_

    groups = []

    #split the points into their respective cluster (all points not put into a cluster are label as -1
    m = max(labels) + 1
    for i in range(m):
        groups.append([])

    for i in range(len(labels)):
        l = labels[i]

        if l == -1:
            continue

        groups[l].append(linePoints[i])

    print("exporting regions to shapefile")

    shpw = shapefile.Writer(projectPath + '/TreeTagger/Regions_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), shapeType=5)
    shpw.field('ID', 'N')

    #run the ConvexHull algorithm to fit polygon around each cluster and then export the polygon to a shapefile
    x = 0
    for group in groups:
        hull = ConvexHull(group)
        hull_vertices = []

        for v in hull.vertices:
            hull_vertices.append(group[v])

        hull_coords = [[topLeft[0] + float(h[0] * imageScale[0]), topLeft[1] - float(h[1]) * imageScale[1]] for h in hull_vertices]
        hull_coords.append(hull_coords[0])
        hull_coords = hull_coords[::-1]

        shpw.poly([hull_coords])
        shpw.record(x)
        x += 1

    shpw.close()
except Exception:
    printError("Error detecting high density regions, make sure the entered settings are correct")

#program has finished, print total run time in seconds
print("time: ", timeit.default_timer() - t)




