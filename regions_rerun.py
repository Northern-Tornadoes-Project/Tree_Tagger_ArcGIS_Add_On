from treetag import *
import sys
import os
import timeit
from datetime import datetime

# timer to measure total runtime
t = timeit.default_timer()

# list of arguments passed from arcgis add in
argv = []
# path to arcgis project folder
project_path = ""

print("Loading arguments")
try:
    # geting project path
    argv = sys.argv[1:]
    project_path = argv[0]
except IndexError:
    print_error("Couldn't load Tree Tagger settings, make sure the entered settings are correct")

try:
    argv = load_args(project_path + "\\TreeTagger\\args.txt")
except FileNotFoundError:
    print_error("Couldn't load Tree Tagger arguments file, make sure the entered settings are correct")

# vector shapefile path
lines_file_path = argv[0]

# coordinate of top left corner
top_left = (float(argv[1]), float(argv[2]))

# coordinate of top left corner
bottom_right = (float(argv[3]), float(argv[4]))

# imagery scale
image_scale = (float(argv[5]), float(argv[6]))

# high density area parameters
parameters = (float(argv[7]), float(argv[8]))

lines = load_lines_shapefile(lines_file_path, image_scale, top_left)

project_pathExtension = "/TreeTagger/Regions_Rerun_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

try:
    print("Calculating high density regions")
    high_density_regions = find_high_density_areas(lines, parameters, top_left, image_scale)

    print("exporting regions to shapefile")
    os.mkdir(project_path + project_pathExtension)
    file_path = project_path + project_pathExtension + '/Regions_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_polygon_shapefile(high_density_regions, file_path)

except Exception:
    print_error("Error detecting high density regions, make sure the entered settings are correct")

# program has finished, print total run time in seconds
print("time: ", timeit.default_timer() - t)
