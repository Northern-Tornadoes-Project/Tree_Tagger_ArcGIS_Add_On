import sys
import traceback


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False


def print_error(msg):
    print("TreeTagger Python Error:\n", msg, traceback.format_exc(), file=sys.stderr)
    exit(1)


def coords_to_pixels(lines, scale, top_left):
    return [[(l[0] - top_left[0])/scale[0], (top_left[1] - l[1])/scale[1], (l[2] - top_left[0])/scale[0],
             (top_left[1] - l[3])/scale[1]] for l in lines]


def coords_points_to_pixels(pts, scale, top_left):
    return [[(p[0] - top_left[0])/scale[0], (top_left[1] - p[1])/scale[1]] for p in pts]


def pixels_to_coords(lines, scale, top_left):
    return [[top_left[0] + l[0] * scale[0], top_left[1] - l[1] * scale[1], top_left[0] + l[2] * scale[0],
             top_left[1] - l[3] * scale[1]] for l in lines]