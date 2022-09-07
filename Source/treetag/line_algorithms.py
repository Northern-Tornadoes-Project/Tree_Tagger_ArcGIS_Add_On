import FastLib
import cv2
import math
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import numpy as np


def extract_lines(image):

    lsd = cv2.ximgproc.createFastLineDetector(do_merge=False)
    return lsd.detect(image)


def join_lines(lines, top_left, bottom_right, image_scale, parameters, grid_size=256.0):

    x = int(math.ceil((bottom_right[0] - top_left[0]) / image_scale[0] / grid_size))
    y = int(math.ceil((top_left[1] - bottom_right[1]) / image_scale[1] / grid_size))

    return FastLib.joinLines(lines, x, y, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4])


def find_lines_in_image(image_data, image_size, image_scale, lines):
    image_lines = []
    new_coord_lines = []

    # if the line is in the image add it to image lines and remove it from coordlines as to not check it twice
    for line in lines:
        if (image_data[1] <= line[0] <= (image_data[1] + image_size[1] * image_scale[0]) and (image_data[2] - image_size[0] * image_scale[1]) <= line[1] <= image_data[2]) or (image_data[1] <= line[2] <= (image_data[1] + image_size[1] * image_scale[0]) and (image_data[2] - image_size[0] * image_scale[1]) <= line[3] <= image_data[2]):

            image_lines.append(line)
        else:
            new_coord_lines.append(line)

    return image_lines, new_coord_lines


def split_lines_into_batches(lines, batch_size):
    line_batches = []

    i = 0
    while i + batch_size < len(lines):
        line_batches.append(lines[i:i + batch_size])
        i += batch_size

    if i + 1 < len(lines):
        line_batches.append(lines[i:i + (len(lines) - i)])

    return line_batches


def build_vectors(direction_grid_parameters, top_left, bottom_right, lines, image_scale):
    grid_vectors = []

    # for each grid direction size (small, med, large)
    for k in range(len(direction_grid_parameters)):
        size = direction_grid_parameters[k][0]
        min_trees = direction_grid_parameters[k][1]

        # get the weighted average grid directions
        x = int(math.ceil((bottom_right[0] - top_left[0]) / size))
        y = int(math.ceil((top_left[1] - bottom_right[1]) / size))

        print(size, min_trees, x, y, (size / image_scale[0]))

        # see Fast lib for more info
        grid_directions = FastLib.averageDirections(lines, (size / image_scale[0]), x, y, 7, min_trees)

        print("back")

        grid_vectors.append([])

        for i in range(y):
            for j in range(x):
                grid_vec = grid_directions[i][j]

                if grid_vec[2] < 0:
                    continue

                grid_vectors[k].append([top_left[0] + j * size + size / 2,
                                        top_left[1] - i * size - size / 2,
                                        top_left[0] + j * size + size / 2 + grid_vec[0] * size / 2,
                                        top_left[1] - i * size - size / 2 - grid_vec[1] * size / 2,
                                        grid_vec[2]])

    return grid_vectors

#5 meters
def refine_lines(lines, image_scale, top_left, bottom_right, min_size=5.0):
    for line in lines:
        if math.sqrt((line[3] - line[1]) ** 2 + (line[2] - line[0]) ** 2) < (min_size / image_scale[0]):
            if line[2] == line[0]:
                line[2] += 1
            slope = float(line[3] - line[1]) / float(line[2] - line[0])
            scale = (min_size / image_scale[0]) / math.sqrt(1 + slope ** 2)
            line[2] = int(line[0] + math.copysign(scale, line[2] - line[0]))
            line[3] = int(line[1] + math.copysign(scale * slope, line[3] - line[1]))

    x = int(math.ceil((bottom_right[0] - top_left[0]) / image_scale[0] / 256.0))
    y = int(math.ceil((top_left[1] - bottom_right[1]) / image_scale[1] / 256.0))

    # joining lines again
    lines = [line[0:4] for line in lines]
    lines = FastLib.joinLines(lines, x, y, float(0.3), float(0.1), float(5), float(5), float(15))
    lines = [[int(l[0]), int(l[1]), int(l[2]), int(l[3])] for l in lines]

    return lines


def find_high_density_areas(lines, parameters, top_left, image_scale):
    line_points = []
    # create a list of all the endpoints of all the lines
    for l in lines:
        line_points.append([int(l[0]), int(l[1])])
        line_points.append([int(l[2]), int(l[3])])

    # create a numpy array for the list
    line_points_np = np.array(line_points)

    # run the dbscan clustering algorithm on the set of points (Note 2 points per line)
    dbscan = DBSCAN(eps=parameters[0], min_samples=(parameters[1] * 2)).fit(line_points_np)

    # get the cluster labels from dbscan
    labels = dbscan.labels_

    groups = []
    hulls = []

    # split the points into their respective cluster (all points not put into a cluster are label as -1
    m = max(labels) + 1
    for i in range(m):
        groups.append([])

    for i in range(len(labels)):
        l = labels[i]

        if l == -1:
            continue

        groups[l].append(line_points[i])

    # run the ConvexHull algorithm to fit polygon around each cluster and then export the polygon to a shapefile
    for group in groups:
        hull = ConvexHull(group)
        hull_vertices = []

        for v in hull.vertices:
            hull_vertices.append(group[v])

        hull_coords = [[top_left[0] + float(h[0] * image_scale[0]), top_left[1] - float(h[1]) * image_scale[1]] for h in hull_vertices]
        hull_coords.append(hull_coords[0])
        hulls.append(hull_coords[::-1])

    return hulls














