import math
import numpy as np
import sys
import os
import cv2
import glob
import tensorflow as tf
from keras_segmentation.models.unet import resnet50_unet
import FastLib
from tqdm.auto import tqdm

def load_model():
    model = resnet50_unet(n_classes=2, input_height=256, input_width=256)
    #model.load_weights(r"F:\python\FCN\checkpoints\resnet50\focal_tversky_plus_wbce_0.02_1.0_0.5_1.0_0.75\weights30_.h5")
    #model.load_weights(r"F:\python\FCN\checkpoints\resnet50\LargeAug1\weights30.h5")
    model.load_weights(os.path.dirname(os.path.realpath(__file__)) + "\\weights30.h5")

    return model

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

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

images = glob.glob(r"F:\Training Data Imagery\Cropped Sections\*.bmp")[58:]
model = load_model()
lsd = cv2.ximgproc.createFastLineDetector(do_merge=False)
X = 94160

for file in tqdm(images, leave=True):
    image = cv2.imread(file)
    mask = np.copy(image)

    #original_image_shape = (image.shape[0], image.shape[1])
    y = int(max(image.shape[0] / 256.0, 1))
    x = int(max(image.shape[1] / 256.0, 1))
    #image = cv2.resize(image, (x * 256, y * 256), cv2.INTER_AREA)

    sections = []
    for i in range(y):
        for j in range(x):
            section = image[i * 256:i * 256 + 256, j * 256:j * 256 + 256]
            section = section.astype(np.float32)
            section = section / 255.0
            sections.append(section)

    sections = np.asarray(sections)

    predictions = model.predict(sections, batch_size=16)

    for i in range(y):
        for j in range(x):
            prediction = np.argmax(predictions[i * x + j], axis=1)
            prediction = np.reshape(prediction, (128, 128, 1))
            prediction = prediction.astype(np.uint8)
            prediction *= 255
            prediction = cv2.resize(prediction, (256, 256), cv2.INTER_CUBIC)
            prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

            mask[i * 256:i * 256 + 256, j * 256:j * 256 + 256] = prediction

    #image = cv2.resize(image, (original_image_shape[1], original_image_shape[0]), cv2.INTER_CUBIC)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    lines = lsd.detect(mask)

    if lines is None:
        continue

    lines = lines.tolist()
    lines = [l[0] for l in lines]

    lines = FastLib.joinLines(lines, x, y, float(0.3), float(0.25), float(25), float(8.5), float(15))
    lines = [[int(l[0]), int(l[1]), int(l[2]), int(l[3])] for l in lines]

    offset = 0

    #border = np.zeros((image.shape[0] + offset * 2, image.shape[1] + offset * 2, image.shape[2]), np.uint8)

    #border[offset:border.shape[0] - offset, offset:border.shape[1] - offset] = image
    #original = np.copy(image)

    for line in tqdm(lines):

        #cv2.line(image, [3055, 730], [3071, 695], [0, 0, 255], 1, cv2.LINE_AA)

        #cv2.imshow("image", image)

        #line = [p + offset for p in line]

        if line[0] == line[2]:
            line[2] += 1

        if line[1] == line[3]:
            line[3] += 1

        mid_point = [round((line[0] + line[2]) / 2.0), round((line[1] + line[3]) / 2.0)]
        angle = math.atan((line[3] - line[1]) / (line[2] - line[0] + 0.00001)) * (180.0 / 3.14159265358979)
        slope = float(line[3] - line[1]) / float(line[2] - line[0])
        nr_slope = -1.0 / slope

        scale = 256.0 / math.sqrt(1 + slope ** 2)

        extended_line = [int(mid_point[0] - scale), int(mid_point[1] - slope * scale),
                         int(mid_point[0] + scale), int(mid_point[1] + slope * scale)]


        # if extended_line[0] < offset:
        #     extended_line[0] = offset
        #     extended_line[1] = int(mid_point[1] - slope * (mid_point[0] - offset))
        # if extended_line[1] < offset or extended_line[1] >= image.shape[0] - offset:
        #     extended_line[1] = int(min(max(offset, extended_line[1]), image.shape[0] - 1 - offset))
        #     extended_line[0] = int(mid_point[0] - (extended_line[1] - mid_point[1]) / slope)
        #
        # if extended_line[2] >= image.shape[1] - offset:
        #     extended_line[2] = image.shape[1] - 1 - offset
        #     extended_line[3] = int(mid_point[1] + slope * (image.shape[1] - 1 - offset - mid_point[0]))
        # if extended_line[3] < 0 or extended_line[3] >= image.shape[0] - offset:
        #     extended_line[3] = int(min(max(offset, extended_line[3]), image.shape[0] - 1 - offset))
        #     extended_line[2] = int(mid_point[0] + (extended_line[3] - mid_point[1]) / slope)

        scale = 256.0 / math.sqrt(1 + nr_slope ** 2)

        skip = False

        box_points = np.array([[int(extended_line[0] - scale), int(extended_line[1] - nr_slope * scale)],
                               [int(extended_line[0] + scale), int(extended_line[1] + nr_slope * scale)],
                               [int(extended_line[2] + scale), int(extended_line[3] + nr_slope * scale)],
                               [int(extended_line[2] - scale), int(extended_line[3] - nr_slope * scale)]], np.int32)

        for p in box_points:
            if p[0] < 0 or p[0] >= image.shape[1] or p[1] < 0 or p[1] >= image.shape[0]:
                skip = True

        if skip:
            continue

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

        box_points_offset = np.array([[p[0] - bounding_box[0][0], p[1] - bounding_box[0][1]] for p in box_points], np.int)

        box_image = image[bounding_box[0][1]:bounding_box[1][1], bounding_box[0][0]:bounding_box[1][0]].copy()

        cv2.line(box_image, [line[0] - bounding_box[0][0], line[1] - bounding_box[0][1]], [line[2] - bounding_box[0][0], line[3] - bounding_box[0][1]], [0, 0, 255], 1, cv2.LINE_AA)

        cv2.imwrite("Z:/tree_direction_full_images/img" + str(X) + ".bmp", box_image)

        mask = np.zeros((bounding_box[1][1] - bounding_box[0][1], bounding_box[1][0] - bounding_box[0][0]), np.uint8)

        cv2.drawContours(mask, [box_points_offset], -1, 255, -1)

        cv2.bitwise_and(box_image, box_image, mask=mask)

        rotated_image = rotate_image(box_image, angle)
        mid_point = [int(rotated_image.shape[1] / 2), int(rotated_image.shape[0] / 2)]

        #print(rotated_image.shape)
        box_image = rotated_image[mid_point[1] - 64:mid_point[1] + 64, mid_point[0] - 256:mid_point[0] + 256]

        box_image = cv2.resize(box_image, (256, 128), interpolation=cv2.INTER_NEAREST)

        #padded_image = np.zeros((256, 256, 3), np.uint8)

        #padded_image[64:192, 0:256] = box_image

        cv2.imwrite("Z:/tree_direction_images/img" + str(X) + ".bmp", box_image)
        X += 1

#cv2.line(original, extended_line[0:2], extended_line[2:4], [255, 0, 255], 3)
#cv2.polylines(original, [box_points], True, [255, 0, 0], 3)

#cv2.imshow("image", original)
#cv2.imshow("test", padded_image)

#cv2.waitKey(0)
