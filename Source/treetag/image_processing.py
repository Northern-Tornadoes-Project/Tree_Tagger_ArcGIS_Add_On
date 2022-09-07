import os
import timeit

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from .utilities import *
import numpy as np
import math


# Rotates an image (angle in degrees) and expands image to avoid cropping
def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origin) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def crop_coord_polygon(coord_points, image_scale, region_scale, top_left, mask):
    # convert polygon points from coords to pixels and resize the polygon to fit fixed scale factor
    scale = [image_scale[0] / region_scale, image_scale[1] / region_scale]

    pixel_points = coords_to_pixels(coord_points, scale, top_left)

    # draw filled polygon on mask image so that it can be bitwise ANDed effectively
    cv2.drawContours(mask, [np.array(pixel_points, np.int)], -1, (255, 255, 255), -1, cv2.LINE_AA)


def extract_scaled_mask(image, mask, region_scale, offset):
    x_offset = 0
    y_offset = 0

    if int(offset[1] / region_scale) == int((offset[1] + image.shape[0]) / region_scale):
        y_offset = 1
    if int(offset[0] / region_scale) == int((offset[0] + image.shape[1]) / region_scale):
        x_offset = 1

    mask_section = mask[int(offset[1] / region_scale):int((offset[1] + image.shape[0]) / region_scale) + y_offset,
                        int(offset[0] / region_scale):int((offset[0] + image.shape[1]) / region_scale) + x_offset]

    mask_section = cv2.resize(mask_section, (image.shape[1], image.shape[0]), cv2.INTER_CUBIC)

    image = cv2.bitwise_and(image, image, mask=mask_section)

    return image


def equalize_image(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)


def batch_line_boxes(image, batch, extended_region, resize_offset):
    batch_images = []

    for i in range(len(batch)):
        line = batch[i]
        # offset the line based on extended border of image
        line = [line[0] + resize_offset[0], line[1] + resize_offset[1], line[2] + resize_offset[0],
                line[3] + resize_offset[1], line[4]]

        # if line is perfectly horizontal / vertical shift it by one pixel (avoids multiplying / dividing by 0)
        if line[0] == line[2]:
            line[2] += 1

        if line[1] == line[3]:
            line[3] += 1

        # mid-point of the line
        mid_point = [round((line[0] + line[2]) / 2.0), round((line[1] + line[3]) / 2.0)]
        # angle of the line
        angle = math.atan((line[3] - line[1]) / (line[2] - line[0] + 0.00001)) * (180.0 / 3.14159265358979)
        # slope of the line
        slope = float(line[3] - line[1]) / float(line[2] - line[0])
        # negative reciprocal slope (perpendicular slope)
        nr_slope = -1.0 / slope

        # scale of x extension to extend line
        scale = resize_offset[0] / math.sqrt(1 + slope ** 2)

        extended_line = [int(mid_point[0] - scale), int(mid_point[1] - slope * scale),
                         int(mid_point[0] + scale), int(mid_point[1] + slope * scale)]

        # scale of x in the perpendicular direction to create box around the tree
        scale = (resize_offset[0] / 4.0) / math.sqrt(1 + nr_slope ** 2)

        skip = False

        # points of the box to crop out section around tree for rotation
        box_points = np.array([[int(extended_line[0] - scale), int(extended_line[1] - nr_slope * scale)],
                               [int(extended_line[0] + scale), int(extended_line[1] + nr_slope * scale)],
                               [int(extended_line[2] + scale), int(extended_line[3] + nr_slope * scale)],
                               [int(extended_line[2] - scale), int(extended_line[3] - nr_slope * scale)]], np.int32)

        # make sure the box fits within the extended image
        for p in box_points:
            # in main image
            if resize_offset[0] < p[0] < image.shape[1] - resize_offset[0] and resize_offset[1] < p[1] < image.shape[0] - \
                    resize_offset[1]:
                continue
            # not in extended image
            if p[0] < 0 or p[0] >= image.shape[1] or p[1] < 0 or p[1] >= image.shape[0]:
                skip = True
                break
            # left side of extended
            elif p[0] < resize_offset[0] and extended_region[1][0] == "None":
                skip = True
                break
            # right side of extended
            elif p[0] > image.shape[1] - resize_offset[0] and extended_region[1][2] == "None":
                skip = True
                break
            # top of extended
            elif p[1] < resize_offset[1] and extended_region[0][1] == "None":
                skip = True
                break
            # bottom of extended
            elif p[1] > image.shape[0] - resize_offset[1] and extended_region[2][1] == "None":
                skip = True
                break
            # top left corner
            elif p[0] < resize_offset[0] and p[1] < resize_offset[1] and extended_region[0][0] == "None":
                skip = True
                break
            # top right corner
            elif p[0] > image.shape[1] - resize_offset[0] and p[1] < resize_offset[1] and extended_region[0][2] == "None":
                skip = True
                break
            # bottom right corner
            elif p[0] > image.shape[1] - resize_offset[0] and p[1] > image.shape[0] - resize_offset[1] and extended_region[2][
                2] == "None":
                skip = True
                break
            # bottom left corner
            elif p[0] < resize_offset[0] and p[1] > image.shape[0] - resize_offset[1] and extended_region[2][0] == "None":
                skip = True

        # if the box doesn't fit within the extended image, mark the line for be skipped later
        if skip:
            batch[i][4] = -1
            continue

        # sort bounding box points so that we can crop out a non-rotated box around the box which surround the tree
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

        # offseting box points so that the top left corner of the box is at the origin
        box_points_offset = np.array([[p[0] - bounding_box[0][0], p[1] - bounding_box[0][1]] for p in box_points],
                                     np.int)

        # cropping out the box surrounding the tree box
        box_image = np.copy(image[bounding_box[0][1]:bounding_box[1][1], bounding_box[0][0]:bounding_box[1][0]])

        # drawing the found line over the tree
        cv2.line(box_image, [line[0] - bounding_box[0][0], line[1] - bounding_box[0][1]],
                 [line[2] - bounding_box[0][0], line[3] - bounding_box[0][1]], [0, 0, 255], 1, cv2.LINE_AA)

        # mask to extract the tree box
        mask = np.zeros((bounding_box[1][1] - bounding_box[0][1], bounding_box[1][0] - bounding_box[0][0]), np.uint8)

        # drawing box to build mask
        cv2.drawContours(mask, [box_points_offset], -1, 255, -1)

        # bitwise AND to extract mask
        cv2.bitwise_and(box_image, box_image, mask=mask)

        # rotating image to make it horizontal
        rotated_image = rotate_image(box_image, angle)

        # cropping out the tree box and stretching it
        mid_point = [int(rotated_image.shape[1] / 2), int(rotated_image.shape[0] / 2)]

        box_image = rotated_image[mid_point[1] - 64:mid_point[1] + 64, mid_point[0] - 256:mid_point[0] + 256]

        box_image = cv2.resize(box_image, (256, 128), interpolation=cv2.INTER_NEAREST)

        # making the tree box image a numpy float32 array
        pImage = np.array(box_image, np.float32)
        # normalizing the image
        pImage /= 255.0
        # adding it to a batch to be processed in parallel
        batch_images.append(pImage)

    return batch_images


def find_surrounding_images(image, image_data, image_scale, all_images, image_sizes, line_extension_size):
    # The tree direction CNN needs to analysis a box around each tree. Since that box could intersect multiple images, we
    # have to know if their exists images which surround the current image being processed (3x3 grid)
    # since some images for events are not in a perfect grid (skewed / not all the same size, etc.) this is a bit of a pain.
    # Also, to save memory the images are loaded one at a time and the bordering section is cropped out
    extended_region = [["None"] * 3,
                       ["None"] * 3,
                       ["None"] * 3]

    for i in range(len(all_images)):
        data = all_images[i]
        width = image_sizes[i][0] * image_scale[0]
        height = image_sizes[i][1] * image_scale[1]

        # top left corner
        if extended_region[0][0] == "None" and width >= line_extension_size and height >= line_extension_size and round(data[1] + width) == round(image_data[1]) and round(data[2] - height) == round(image_data[2]):
            extended_region[0][0] = data[0]

        # top
        elif extended_region[0][1] == "None" and height >= line_extension_size and round(data[1]) == round(image_data[1]) and round(data[2] - height) == round(image_data[2]):
            extended_region[0][1] = data[0]

        # top right corner
        elif extended_region[0][2] == "None" and width >= line_extension_size and height >= line_extension_size and round(data[1]) == round(image_data[1] + image.shape[1] * image_scale[0]) and round(data[2] - height) == round(image_data[2]):
            extended_region[0][2] = data[0]

        # left
        elif extended_region[1][0] == "None" and width >= line_extension_size and round(data[1] + width) == round(image_data[1]) and round(data[2]) == round(image_data[2]):
            extended_region[1][0] = data[0]

        # right
        elif extended_region[1][2] == "None" and width >= line_extension_size and round(data[1]) == round(image_data[1] + image.shape[1] * image_scale[0]) and round(data[2]) == round(image_data[2]):
            extended_region[1][2] = data[0]

        # bottom left corner
        elif extended_region[2][0] == "None" and width >= line_extension_size and height >= line_extension_size and round(data[1] + width) == round(image_data[1]) and round(data[2]) == round(image_data[2] - image.shape[0] * image_scale[1]):
            extended_region[2][0] = data[0]

        # bottom
        elif extended_region[2][1] == "None" and height >= line_extension_size and round(data[1]) == round(image_data[1]) and round(data[2]) == round(image_data[2] - image.shape[0] * image_scale[1]):
            extended_region[2][1] = data[0]

        # bottom right corner
        elif extended_region[2][2] == "None" and width >= line_extension_size and height >= line_extension_size and round(data[1]) == round(image_data[1] + image.shape[1] * image_scale[0]) and round(data[2]) == round(image_data[2] - image.shape[0] * image_scale[1]):
            extended_region[2][2] = data[0]

    return extended_region


def crop_extended_region(image, extended_region, resize_offset):
    original_size = image.shape

    # image extended on all sizes by resize offset
    extended_image = np.zeros((original_size[0] + 2 * resize_offset[1], original_size[1] + 2 * resize_offset[0], 3), np.uint8)
    # moving image to center of extended image
    extended_image[resize_offset[1]:resize_offset[1] + original_size[0],
                   resize_offset[0]:resize_offset[0] + original_size[1]] = image

    # top left corner
    if extended_region[0][0] != "None":
        image = cv2.imread(extended_region[0][0], cv2.IMREAD_COLOR)
        extended_image[0:resize_offset[1], 0:resize_offset[0]] = image[image.shape[0] - resize_offset[1]:image.shape[0],
                                                                       image.shape[1] - resize_offset[0]:image.shape[1]]

    # top
    if extended_region[0][1] != "None":
        image = cv2.imread(extended_region[0][1], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (original_size[1], image.shape[0]), cv2.INTER_CUBIC)
        extended_image[0:resize_offset[1], resize_offset[0]:resize_offset[0] + image.shape[1]] = image[image.shape[0] - resize_offset[1]:image.shape[0],0:image.shape[1]]

    # top right corner
    if extended_region[0][2] != "None":
        image = cv2.imread(extended_region[0][2], cv2.IMREAD_COLOR)
        extended_image[0:resize_offset[1], extended_image.shape[1] - resize_offset[0]:extended_image.shape[1]] = image[image.shape[0] - resize_offset[1]:image.shape[0], 0:resize_offset[0]]

    # left
    if extended_region[1][0] != "None":
        image = cv2.imread(extended_region[1][0], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (image.shape[1], original_size[0]), cv2.INTER_CUBIC)
        extended_image[resize_offset[1]:resize_offset[1] + image.shape[0], 0:resize_offset[0]] = image[0:image.shape[0], image.shape[1] - resize_offset[0]:image.shape[1]]

    # right
    if extended_region[1][2] != "None":
        image = cv2.imread(extended_region[1][2], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (image.shape[1], original_size[0]), cv2.INTER_CUBIC)
        extended_image[resize_offset[1]:resize_offset[1] + image.shape[0],
                       extended_image.shape[1] - resize_offset[0]:extended_image.shape[1]] = image[0:image.shape[0], 0:resize_offset[0]]

    # bottom left corner
    if extended_region[2][0] != "None":
        image = cv2.imread(extended_region[2][0], cv2.IMREAD_COLOR)
        extended_image[extended_image.shape[0] - resize_offset[1]:extended_image.shape[0], 0:resize_offset[0]] = image[0:resize_offset[1], image.shape[1] - resize_offset[0]:image.shape[1]]

    # bottom
    if extended_region[2][1] != "None":
        image = cv2.imread(extended_region[2][1], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (original_size[1], image.shape[0]), cv2.INTER_CUBIC)
        extended_image[extended_image.shape[0] - resize_offset[1]:extended_image.shape[0],
                       resize_offset[0]:resize_offset[0] + image.shape[1]] = image[0:resize_offset[1], 0:image.shape[1]]

    # bottom right corner
    if extended_region[2][2] != "None":
        image = cv2.imread(extended_region[2][2], cv2.IMREAD_COLOR)
        extended_image[extended_image.shape[0] - resize_offset[1]:extended_image.shape[0],
                       extended_image.shape[1] - resize_offset[0]:extended_image.shape[1]] = image[0:resize_offset[1], 0:resize_offset[0]]

    return extended_image

