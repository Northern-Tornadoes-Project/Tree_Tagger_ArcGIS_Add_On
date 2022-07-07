import math
import multiprocessing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import cv2
import tensorflow.keras as keras
from keras_segmentation.models.unet import resnet50_unet
from keras_segmentation.predict import predict
from glob import glob
from tqdm import tqdm
import sys
from osgeo import gdal, osr
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import FastLib
import timeit
import shapefile
import os
from datetime import datetime


def load_model():
    model = resnet50_unet(n_classes=2, input_height=256, input_width=256)
    #model.load_weights(r"F:\python\FCN\checkpoints\resnet50\focal_tversky_plus_wbce_0.02_1.0_0.5_1.0_0.75\weights30_.h5")
    model.load_weights(r"F:\python\FCN\checkpoints\resnet50\LargeAug1\weights30.h5")

    return model


def get_prediction(model, X):
    prediction = model.predict_segmentation(X)

    output = np.argmax(prediction, axis=1)
    output = np.reshape(output, (128, 128, 1))
    output = output.astype(np.uint8)
    output *= 255

    # output = np.zeros((128, 128, 1), np.uint8)
    #
    # for i in range(128):
    #     for j in range(128):
    #         p = prediction[i * 128 + j]
    #
    #         if abs(p[0]) < abs(p[1]):
    #             output[i, j] = 255

    output = cv2.resize(output, (256, 256), cv2.INTER_CUBIC)

    return output


def dist2(v, w):
    return (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2


def distToSegment(l, p):
    v = (l[0], l[1])
    w = (l[2], l[3])

    l2 = dist2(v, w)
    if l2 == 0:
        return dist2(p, v)

    t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2
    t = max(0.0, min(1.0, t))

    return dist2(p, (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1])))


t = timeit.default_timer()

print("Loading arguments")
argv = sys.argv[1:]

projectPath = os.path.dirname(argv[0])

print(projectPath)

imageScale = (float(argv[1]), float(argv[2]))

topLeft = (float(argv[3]), float(argv[4]))

bottomRight = (float(argv[5]), float(argv[6]))

polygon = bool(int(argv[7]))

imgArgs = argv[8:]

imagesData = []

for i in range(0, len(imgArgs), 3):
    imagesData.append((imgArgs[i], float(imgArgs[i + 1]), float(imgArgs[i + 2])))

print("Loading machine learning model")
model = load_model()

# if polygon:
#     print("Loading selected image")
#     image = cv2.imread(r"%s" % url)
#
#     print("Cropping out polygon")
#     pointArgs = list(map(int, argv[9:]))
#     points = []
#
#     for i in range(0, len(pointArgs), 2):
#         points.append([pointArgs[i], pointArgs[i + 1]])
#
#     points = np.array(points)
#
#     args[2] = math.ceil(args[2] / 256) * 256
#     args[3] = math.ceil(args[3] / 256) * 256
#
#     image = image[int(args[1]):int(args[1] + args[3]), int(args[0]):int(args[0] + args[2])].copy()
#     y = int(args[3] / 256)
#     x = int(args[2] / 256)
#
#     points = points - points.min(axis=0)
#
#     mask = np.zeros(image.shape[:2], np.uint8)
#
#     cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
#
#     image = cv2.bitwise_and(image, image, mask=mask)

print("Calculating predictions and detecting lines")
allLines = []

for imgData in tqdm(imagesData):
    image = cv2.imread(imgData[0], cv2.IMREAD_COLOR)
    print(image.shape)
    original_image_shape = (image.shape[0], image.shape[1])
    y = int(original_image_shape[0] / 256.0)
    x = int(original_image_shape[1] / 256.0)
    image = cv2.resize(image, (x*256, y*256), cv2.INTER_AREA)
    print(image.shape)

    sections = []
    for i in range(y):
        for j in range(x):
            section = image[i * 256:i * 256 + 256, j * 256:j * 256 + 256]
            section = section.astype(np.float32)
            section = section/255.0
            sections.append(section)

    sections = np.asarray(sections)

    predictions = model.predict(sections, batch_size=16)

    for i in range(y):
        for j in range(x):
            prediction = np.argmax(predictions[i*y + j], axis=1)
            prediction = np.reshape(prediction, (128, 128, 1))
            prediction = prediction.astype(np.uint8)
            prediction *= 255
            prediction = cv2.resize(prediction, (256, 256), cv2.INTER_CUBIC)
            prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

            image[i * 256:i * 256 + 256, j * 256:j * 256 + 256] = prediction

    image = cv2.resize(image, (original_image_shape[1], original_image_shape[0]), cv2.INTER_CUBIC)
    print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # print("Denoising Image")
    # cv2.fastNlMeansDenoising(filteredImage, filteredImage, 50, 7, 21)
    #
    # print("Applying binary filter")
    # cv2.threshold(filteredImage, 100, 255, cv2.THRESH_BINARY, filteredImage)
    #
    # image = cv2.cvtColor(filteredImage, cv2.COLOR_GRAY2BGR)
    #
    # print("Applying blur filter")
    # filteredImage = cv2.blur(filteredImage, (2, 2))

    lsd = cv2.ximgproc.createFastLineDetector(do_merge=False)

    #lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines = lsd.detect(image)

    if lines is not None:
        lines = lines.tolist()

        lines = [l[0] for l in lines]

        offset = ((imgData[1] - topLeft[0]) / imageScale[0], (topLeft[1] - imgData[2]) / imageScale[1])

        #converting coords to pixels
        lines = [[l[0] + offset[0], l[1] + offset[1], l[2] + offset[0], l[3] + offset[1]] for l in lines]

        allLines.extend(lines)

print("Joining lines")

x = int(math.ceil((bottomRight[0] - topLeft[0]) / imageScale[0] / 256.0))
y = int(math.ceil((topLeft[1] - bottomRight[1]) / imageScale[1] / 256.0))

#print(x, y)

allLines = FastLib.joinLines(allLines, x, y)

print("Exporting lines to shape files")
#converting pixels to coords
coordLines = [[topLeft[0] + l[0] * imageScale[0], topLeft[1] - l[1] * imageScale[1], topLeft[0] + l[2] * imageScale[0], topLeft[1] - l[3] * imageScale[1]] for l in allLines]

if not os.path.exists(projectPath + '/TreeTagger'):
    os.makedirs(projectPath + '/TreeTagger')

shpw = shapefile.Writer(projectPath + '/TreeTagger/Lines_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), shapeType=3)
shpw.field('ID', 'C')

x = 0
for l in coordLines:
    shpw.line([[[l[0], l[1]], [l[2], l[3]]]])
    shpw.record(x)
    x += 1

shpw.close()

# alpha = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY, alpha)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
# image[:, :, 3] = alpha

print("Calculating high density regions")
linePoints = []
for l in allLines:
    linePoints.append([int(l[0]), int(l[1])])
    linePoints.append([int(l[2]), int(l[3])])

linePointsnp = np.array(linePoints)

dbscan = DBSCAN(eps=300, min_samples=50).fit(linePointsnp)

labels = dbscan.labels_

groups = []

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
shpw.field('ID', 'C')

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

    #cv2.drawContours(image, [np.array(hull_vertices)], -1, (255, 0, 255, 120), -1, cv2.LINE_AA)

# for l in lines:
#     cv2.line(image, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255, 255), 2, cv2.LINE_AA)

# for i in range(x):
#     cv2.line(image, (i*256, 0), (i*256, y*256), (0, 0, 0, 255), 1, cv2.LINE_AA)
#
# cv2.line(image, (x * 256 - 1, 0), (x * 256 - 1, y * 256), (0, 0, 0, 255), 1, cv2.LINE_AA)
#
# for i in range(y):
#     cv2.line(image, (0, i*256), (x*256, i*256), (0, 0, 0, 255), 1, cv2.LINE_AA)
#
# cv2.line(image, (0, y * 256 - 1), (x * 256, y * 256 - 1), (0, 0, 0, 255), 1, cv2.LINE_AA)

# cv2.imwrite(r"D:\arcGIS\TreeTagger\mask.tif", image)
#
# print("Georeferencing prediction")
# src_filename = "D:/arcGIS/TreeTagger/mask.tif"
# dst_filename = "D:/arcGIS/TreeTagger/TreeMask.tif"
#
# # Opens source dataset
# src_ds = gdal.Open(src_filename)
# format = "GTiff"
# driver = gdal.GetDriverByName(format)
#
# # Open destination dataset
# dst_ds = driver.CreateCopy(dst_filename, src_ds, 0)
#
# # Specify raster location through geotransform array
# # (uperleftx, scalex, skewx, uperlefty, skewy, scaley)
# # Scale = size of one pixel in units of raster projection
# # this example below assumes 100x100
# gt = [args[4], args[6], 0, args[5], 0, args[6] * -1]
#
# # Set location
# dst_ds.SetGeoTransform(gt)
#
# # Get raster projection
# epsg = 26912
# srs = osr.SpatialReference()
# srs.ImportFromEPSG(epsg)
# dest_wkt = srs.ExportToWkt()
#
# # Set projection
# dst_ds.SetProjection(dest_wkt)
#
# # Close files
# dst_ds = None
# src_ds = None

print("time: ", timeit.default_timer() - t)


