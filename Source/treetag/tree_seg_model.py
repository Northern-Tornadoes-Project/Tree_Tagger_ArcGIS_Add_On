import os
import cv2
import numpy as np
from .image_processing import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras_segmentation.models.unet import resnet50_unet


class TreeSegModel(object):

    def __init__(self, weights_path):
        self.model = resnet50_unet(n_classes=2, input_height=256, input_width=256)
        self.model.load_weights(weights_path)

    def predict_and_interpret(self, image, image_scale, equalize=False, batch_size=16):
        resize_factor = (1.0, 1.0)

        # since the FCN model was trained on 5cm imagery, it expects a tree to be a certain size in terms of pixels and look
        # certain ways, which wouldn't be the case with higher quality imagery
        # as such the image should be resized to match 5cm imagery better
        if image_scale[0] != 0.05 or image_scale[1] != 0.05:
            resize_factor = (image_scale[0] / 0.05, image_scale[1] / 0.05)
            image = cv2.resize(image,
                               (int(image.shape[1] * resize_factor[0]), int(image.shape[0] * resize_factor[1])),
                               cv2.INTER_CUBIC)

        resized_shape = (image.shape[0], image.shape[1])

        # image is again resized so that it is a multiple of 256 as required by the FCN
        y = int(max(image.shape[0] / 256.0, 1))
        x = int(max(image.shape[1] / 256.0, 1))
        image = cv2.resize(image, (x * 256, y * 256), cv2.INTER_AREA)

        if equalize:
            image = equalize_image(image)

        # split the image into 256x256 sections to feed into the FCN model
        # image pixels are also converted to floats and normalized
        sections = []
        for i in range(y):
            for j in range(x):
                section = image[i * 256:i * 256 + 256, j * 256:j * 256 + 256]
                section = section.astype(np.float32)
                section = section / 255.0
                sections.append(section)

        # needs to be numpy array
        sections = np.asarray(sections)

        # get the FCN model to make predictions for each 256x256 section
        predictions = self.model.predict(sections, batch_size=batch_size)

        # convert predictions into appropriate pixels and stitch image back together
        for i in range(y):
            for j in range(x):
                prediction = np.argmax(predictions[i * x + j], axis=1)
                prediction = np.reshape(prediction, (128, 128, 1))
                prediction = prediction.astype(np.uint8)
                prediction *= 255
                prediction = cv2.resize(prediction, (256, 256), cv2.INTER_CUBIC)
                prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

                image[i * 256:i * 256 + 256, j * 256:j * 256 + 256] = prediction

        # resize image back to original size (after being resized to 5cm imagery)
        image = cv2.resize(image, (resized_shape[1], resized_shape[0]), cv2.INTER_CUBIC)

        # convert to gray scale as color is no longer required
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image, resize_factor
