from .utilities import *
import shapefile
import csv


def load_args(file_path):
    # reading arguments file created by arcgis add in
    args_file = open(file_path)
    # arguments are separated using | since it can't be put in a file name
    return args_file.readline().split("|")


def load_lines_shapefile(file_path, image_scale, top_left):
    sfr = shapefile.Reader(file_path)
    shapes = sfr.shapes()

    if shapes[0].shapeType != shapefile.POLYLINE:
        print_error("Invalid shape file type, make sure you select a polyline shape file")

    lines = [[x.points[0][0], x.points[0][1], x.points[1][0], x.points[1][1]] for x in shapes]

    return coords_to_pixels(lines, image_scale, top_left)


def load_vector_shapefile(file_path, image_scale, top_left):
    sfr = shapefile.Reader(file_path)
    shapes = sfr.shapes()

    if shapes[0].shapeType != shapefile.POLYLINE:
        print_error("Invalid shape file type, make sure you select a polyline shape file")

    records = sfr.records()
    lines = [[x.points[0][0], x.points[0][1], x.points[1][0], x.points[1][1]] for x in shapes]

    lines = coords_to_pixels(lines, image_scale, top_left)

    for i in range(len(lines)):
        lines[i].append(records[i][0])

    return lines


def export_line_shapefile(lines, scale, top_left, file_path):
    # converting pixels to coords
    coord_lines = pixels_to_coords(lines, scale, top_left)

    shpw = shapefile.Writer(file_path, shapeType=shapefile.POLYLINE)
    shpw.field('ID', 'N')

    x = 0
    for l in coord_lines:
        shpw.line([[[l[0], l[1]], [l[2], l[3]]]])
        # shape file must have some kind of record (label) for each shape
        shpw.record(x)
        x += 1

    shpw.close()


def export_vector_shapefile(lines, scale, top_left, file_path):
    # converting pixels to coords
    coord_lines = pixels_to_coords(lines, scale, top_left)

    shpw = shapefile.Writer(file_path, shapeType=shapefile.POLYLINE)
    shpw.field('Confidence', 'F', decimal=6)

    for i in range(len(coord_lines)):
        l = coord_lines[i]
        shpw.line([[[l[0], l[1]], [l[2], l[3]]]])
        # shape file must have some kind of record (label) for each shape
        shpw.record(lines[i][4])

    shpw.close()


def export_direction_shapefile(grid_vectors, file_path):
    shpw = shapefile.Writer(file_path, shapeType=shapefile.POLYLINE)
    shpw.field('Consistency', 'F', decimal=6)
    shpw.field('Scale', 'N')

    for i in range(len(grid_vectors)):
        for l in grid_vectors[i]:
            shpw.line([[[l[0], l[1]], [l[2], l[3]]]])
            shpw.record(Consistency=l[4], Scale=(i + 1))

    shpw.close()


def export_polygon_shapefile(polygons, file_path):
    shpw = shapefile.Writer(file_path, shapeType=shapefile.POLYGON)
    shpw.field('ID', 'N')

    x = 0
    for p in polygons:
        shpw.poly([p])
        # shape file must have some kind of record (label) for each shape
        shpw.record(x)
        x += 1

    shpw.close()


def export_direction_csv(grid_vectors, file_path):
    header = ['x1', 'y1', 'x2', 'y2', 'consistency', 'scale']
    file = open(file_path, 'w', newline='')
    csv_writer = csv.writer(file)

    csv_writer.writerow(header)

    for i in range(len(grid_vectors)):
        for l in grid_vectors[i]:
            csv_writer.writerow([l[0], l[1], l[2], l[3], l[4], (i + 1)])

    file.close()
