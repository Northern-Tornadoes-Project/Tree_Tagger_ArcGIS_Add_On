import statistics

import FastLib
import cv2
import math
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from scipy.spatial import ConvexHull
import numpy as np
np.seterr(all="ignore")

def extract_lines(image):

    lsd = cv2.ximgproc.createFastLineDetector(do_merge=False)
    return lsd.detect(image)


def join_lines(lines, top_left, bottom_right, image_scale, parameters, grid_size=256.0):

    x = int(math.ceil((bottom_right[0] - top_left[0]) / image_scale[0] / grid_size))
    y = int(math.ceil((top_left[1] - bottom_right[1]) / image_scale[1] / grid_size))

    return FastLib.joinLines(lines, x, y, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])


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


def kmedoid_cluster(lines, grid_size, x, y, extrapolation_req, min_trees):

    angle_threshold = math.radians(40)
    average_directions_grid = [[[1.0e10, 1.0e10, -1.0] for j in range(x)] for i in range(y)]
    line_grid = [[[] for j in range(x)] for i in range(y)]

    #sort lines into grid
    for line in lines:
        mid = [(line[0] + line[2]) / 2, (line[1] + line[3]) / 2]

        line_grid[int(mid[1] / grid_size)][int(mid[0] / grid_size)].append(line)

    for i in range(y):
        for j in range(x):
            grid_square = line_grid[i][j]
            if len(grid_square) < min_trees:
                continue

            vec_sum = [0.0, 0.0]
            total = 0.0
            unit_vecs = []
            unit_x = []
            unit_y = []

            #computer unit vectors, sum and total
            for line in grid_square:
                line_vec = [line[2] - line[0], line[3] - line[1]]

                mag = math.sqrt(line_vec[0]**2 + line_vec[1]**2)
                unit_vec = [line_vec[0] / mag, line_vec[1] / mag]
                unit_x.append(unit_vec[0])
                unit_y.append(unit_vec[1])

                unit_vecs.append(unit_vec)

                vec_sum[0] += unit_vec[0] * line[4]
                vec_sum[1] += unit_vec[1] * line[4]
                total += line[4]

            #compute consistency score
            wavg_vec = [vec_sum[0] / total, vec_sum[1] / total]
            wavg_mag = math.sqrt(wavg_vec[0]**2 + wavg_vec[1]**2)

            kmedoids = KMedoids(n_clusters=2, method='pam', init='k-medoids++', max_iter=1000).fit(np.asarray(unit_vecs))

            medoid_A = kmedoids.cluster_centers_[0]
            medoid_B = kmedoids.cluster_centers_[1]

            angle = math.acos(max(min(medoid_A[0] * medoid_B[0] + medoid_A[1] * medoid_B[1], 1.0), -1.0))

            sum_B = np.sum(kmedoids.labels_)

            if angle >= angle_threshold and 0.35 <= (sum_B/kmedoids.labels_.size) <= 0.65:
                average_directions_grid[i][j] = [[medoid_A[0], medoid_A[1], wavg_mag], [medoid_B[0], medoid_B[1], wavg_mag]]

            else:
                median_vec = [statistics.median(unit_x), statistics.median(unit_y)]
                mag = math.hypot(median_vec[0], median_vec[1])
                average_directions_grid[i][j] = [median_vec[0] / mag, median_vec[1] / mag, wavg_mag]

            # #angle difference is too small, average medoids
            # if angle <= angle_threshold:
            #     medoid_avg = [(medoid_A[0] + medoid_B[0]) / 2.0, (medoid_A[1] + medoid_B[1]) / 2.0]
            #     mag = math.sqrt(medoid_avg[0]**2 + medoid_avg[1]**2)
            #
            #     average_directions_grid[i][j] = [medoid_avg[0] / mag, medoid_avg[1] / mag, wavg_mag]
            #
            # else:
            #     sum_B = np.sum(kmedoids.labels_)
            #
            #     #create two vectors
            #     if 0.4 <= (sum_B/kmedoids.labels_.size) <= 0.6:
            #         average_directions_grid[i][j] = [[medoid_A[0], medoid_A[1], wavg_mag], [medoid_B[0], medoid_B[1], wavg_mag]]
            #
            #     #take bigger cluster's medoid
            #     else:
            #         sum_A = kmedoids.labels_.size - sum_B
            #
            #         if sum_A > sum_B:
            #             average_directions_grid[i][j] = [medoid_A[0], medoid_A[1], wavg_mag]
            #
            #         else:
            #             average_directions_grid[i][j] = [medoid_B[0], medoid_B[1], wavg_mag]

    return average_directions_grid


def build_vectors(direction_grid_parameters, interpolation, top_left, bottom_right, lines, image_scale):
    grid_vectors = []

    # for each grid direction size
    for k in range(len(direction_grid_parameters)):
        size = direction_grid_parameters[k][0]
        min_trees = direction_grid_parameters[k][1]

        # get the weighted average grid directions
        x = int(math.ceil((bottom_right[0] - top_left[0]) / size))
        y = int(math.ceil((top_left[1] - bottom_right[1]) / size))

        #print(size, min_trees, x, y, (size / image_scale[0]))

        # see Fast lib for more info
        grid_directions = []

        print(interpolation)

        if interpolation == "W Average":
            grid_directions = FastLib.averageDirections(lines, (size / image_scale[0]), x, y, 7, min_trees)

        elif interpolation == "XY Median":
            grid_directions = FastLib.medianDirections(lines, (size / image_scale[0]), x, y, 7, min_trees)

        elif interpolation == "Medoid":
            grid_directions = FastLib.medoidDirections(lines, (size / image_scale[0]), x, y, 7, min_trees)

        elif interpolation == "2-Cluster":
            grid_directions = kmedoid_cluster(lines, (size / image_scale[0]), x, y, 7, min_trees)

        grid_vectors.append([])

        for i in range(y):
            for j in range(x):
                grid_vec = grid_directions[i][j]

                if type(grid_vec[0]) == list:
                    for vec in grid_vec:
                        if vec[2] < 0:
                            continue

                        dx2 = vec[0] * size / 4.0
                        dy2 = vec[1] * size / 4.0

                        # scale and center vectors in grid squares
                        grid_vectors[k].append([top_left[0] + j * size + size / 2 - dx2,
                                                top_left[1] - i * size - size / 2 + dy2,
                                                top_left[0] + j * size + size / 2 + dx2,
                                                top_left[1] - i * size - size / 2 - dy2,
                                                vec[2]])

                else:
                    if grid_vec[2] < 0:
                        continue

                    dx2 = grid_vec[0]*size / 4.0
                    dy2 = grid_vec[1]*size / 4.0

                    #scale and center vectors in grid squares
                    grid_vectors[k].append([top_left[0] + j * size + size / 2 - dx2,
                                            top_left[1] - i * size - size / 2 + dy2,
                                            top_left[0] + j * size + size / 2 + dx2,
                                            top_left[1] - i * size - size / 2 - dy2,
                                            grid_vec[2]])

    return grid_vectors


#5 meters
def refine_lines(lines, image_scale, top_left, bottom_right, parameters, min_size=5.0):
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
    lines = FastLib.joinLines(lines, x, y, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])
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














