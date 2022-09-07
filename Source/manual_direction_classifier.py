import cv2
import glob
import random
from sklearn.utils import shuffle
from tqdm import tqdm

image_files = glob.glob("Z:/tree_direction_images/*.bmp")
full_files = glob.glob("Z:/tree_direction_full_images/*.bmp")
left = glob.glob("Z:/labelled_tree_directions/left/*.bmp")
right = glob.glob("Z:/labelled_tree_directions/right/*.bmp")
inc = glob.glob("Z:/labelled_tree_directions/inc/*.bmp")

previous = left + right + inc

for file in tqdm(previous):
    split = file.split('\\')
    name = split[len(split) - 1]
    image_files.remove("Z:/tree_direction_images\\" + name)
    full_files.remove("Z:/tree_direction_full_images\\" + name)

image_files, full_files = shuffle(image_files, full_files, random_state=0)

for i in range(len(image_files)):
    image_file = image_files[i]
    full_file = full_files[i]
    split = image_file.split('\\')
    name = split[len(split) - 1]
    image = cv2.imread(image_file)
    big_image = cv2.resize(image, (1024, 512), cv2.INTER_NEAREST)
    full_image = cv2.imread(full_file)
    cv2.imshow("image", big_image)
    cv2.imshow("full_image", full_image)

    while True:
        key = cv2.waitKeyEx()

        #left
        if key == 2424832:
            cv2.imwrite("Z:/labelled_tree_directions/left/" + name, image)
            print("left")
            break

        #right
        if key == 2555904:
            cv2.imwrite("Z:/labelled_tree_directions/right/" + name, image)
            print("right")
            break

        #up (inc)
        if key == 2490368:
            cv2.imwrite("Z:/labelled_tree_directions/inc/" + name, image)
            print("inc")
            break


