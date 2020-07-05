#!/usr/local/bin/python3
# Based on skeleton code by D. Crandall, Oct 2019

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio


# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    return sqrt(filtered_y ** 2)


# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range(int(max(y - int(thickness / 2), 0)), int(min(y + int(thickness / 2), image.size[1] - 1))):
            image.putpixel((x, t), color)
    return image


# main program
#
(input_filename, gt_row, gt_col) = sys.argv[1:]
gt_row = int(gt_row)
gt_col = int(gt_col)

# load in image
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)

rows, columns = edge_strength.shape


def part1():
    # Calculating emissions
    edge_strength_sum = edge_strength.sum(axis=0)
    edge_strength_divide = edge_strength / edge_strength_sum
    a_ridge = argmax(edge_strength_divide, axis=0)

    # output answer
    # Part 1
    imageio.imwrite("output_simple.jpg", draw_edge(input_image, a_ridge, (0, 0, 255), 5))


def part2():
    # Calculate emissions for column 1
    edge_strength.dtype = float64
    edge_strength_sum = edge_strength.sum(axis=0)
    edge_strength_divide = edge_strength / edge_strength_sum
    edge_strength_divide[:, 0] = edge_strength_divide[:, 0] / rows

    # Viterbi
    viterbi = edge_strength_divide * 100
    final_maximum_list = []

    for column in range(1, columns):
        list_of_maximums = []

        for row1 in range(rows):
            values_to_max = []
            temp1 = []

            for row in range(rows):
                if abs(row1 - row) < 3:
                    values_to_max.append(0.15 * viterbi[row][column - 1])

                else:
                    values_to_max.append(((1 - 0.9) / (rows - 6)) * viterbi[row][column - 1])

            viterbi[row1][column] = viterbi[row1][column] * max(values_to_max)
            list_of_maximums.append(values_to_max.index(max(values_to_max)))

        final_maximum_list.append(list_of_maximums)

    viterbi_last = argmax(viterbi[:, -1], axis=0)
    next_index = viterbi_last
    edge_index = [viterbi_last]
    for i in range(columns - 2, 0, -1):
        edge_index.append(final_maximum_list[i][next_index])
        next_index = final_maximum_list[i][next_index]

    # Reverse the list formed
    edge_index = edge_index[::-1]
    input_image = Image.open(input_filename)
    imageio.imwrite("output_map.jpg", draw_edge(input_image, edge_index, (255, 0, 0), 5))


def part3(viterbi, columns):
    viterbi = viterbi * 100
    final_maximum_list = []
    for column in range(1, columns):
        list_of_maximums = []
        for row1 in range(rows_global):
            values_to_max = []
            curr = row1
            if abs(argmax(edge_strength[:, gt_col], axis=0) - gt_row) > 10:
                curr = gt_row
            for row in range(rows_global):
                if abs(curr - row) < 8:
                    values_to_max.append(0.1 * viterbi[row][column - 1])
                else:
                    values_to_max.append(((1 - 0.8) / (rows_global - 15)) * viterbi[row][column - 1])
            viterbi[row1][column] = viterbi[row1][column] * max(values_to_max)
            list_of_maximums.append(values_to_max.index(max(values_to_max)))

        final_maximum_list.append(list_of_maximums)

    viterbi_last = argmax(viterbi[:, -1], axis=0)
    next_index = viterbi_last
    edge_index = [viterbi_last]
    for i in range(columns - 2, 0, -1):
        edge_index.append(final_maximum_list[i][next_index])
        next_index = final_maximum_list[i][next_index]

    # Reverse the list formed
    edge_index = edge_index[::-1]
    return edge_index


# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

# output answer
# Part 1
part1()

# Part 2
part2()



rows_global, columns_global = edge_strength.shape

edge_strength.dtype = float64
edge_strength_sum = edge_strength.sum(axis=0)
edge_strength_divide = edge_strength / edge_strength_sum
edge_strength_divide[:, gt_col] = 0
edge_strength_divide[gt_row][gt_col] = 1

if gt_col == 0:
    edge_index = part3(edge_strength_divide, columns_global)
    input_image = Image.open(input_filename)
    imageio.imwrite("output_human.jpg", draw_edge(input_image, edge_index, (0, 255, 0), 5))

else:
    edge_strength_divide1 = edge_strength_divide[:, 0:gt_col + 1]
    edge_strength_divide1 = flip(edge_strength_divide1, axis=1)
    edge_strength_divide2 = edge_strength_divide[:, gt_col:columns_global + 1]
    rows_1, columns_1 = edge_strength_divide1.shape
    edge_index1 = part3(edge_strength_divide1, columns_1)
    edge_index1 = flip(edge_index1)
    edge_index1 = list(edge_index1)

    rows_2, columns_2 = edge_strength_divide2.shape
    edge_index2 = part3(edge_strength_divide2, columns_2)
    edge_index2 = list(edge_index2)

    edge_index1 = edge_index1[:-1]
    edge_index = edge_index1 + edge_index2
    input_image = Image.open(input_filename)
    imageio.imwrite("output_human.jpg", draw_edge(input_image, edge_index, (0, 255, 0), 5))
