import cv2
import numpy as np
import thinning
from .image_processing import *
np.seterr(all="ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from keras_segmentation.models.unet import resnet50_unet
import segmentation_models as sm
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class TreeSegModel(object):

    def __init__(self, weights_path):
        self.model = sm.Unet("vgg16", input_shape=(256, 256, 3), classes=2, encoder_weights=None)
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
        mask = np.zeros(image.shape, np.uint8)

        if equalize:
            image = equalize_image(image)

        rotM = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

        for s in range(2):
            q = 1
            g = 128
            if s == 0:
                q = 0
            sections = []
            for r in range(4):

                if r == 0:
                    for i in range(y - q):
                        for j in range(x - q):
                            section = image[i * 256 + g * s:i * 256 + 256 + g * s, j * 256 + g * s:j * 256 + 256 + g * s]
                            section = section.astype(np.float32)
                            section = section / 255.0
                            sections.append(section)

                    sections_np = np.asarray(sections)
                    # predictions = []
                    # predictions.extend(self.model.predict(sections_np[0:int(len(sections_np) / 4)], batch_size=batch_size))
                    # predictions.extend(self.model.predict(sections_np[int(len(sections_np) / 4):2 * int(len(sections_np) / 4)], batch_size=batch_size))
                    # predictions.extend(self.model.predict(sections_np[2 * int(len(sections_np) / 4):3 * int(len(sections_np) / 4)], batch_size=batch_size))
                    # predictions.extend(self.model.predict(sections_np[2 * int(len(sections_np) / 4):], batch_size=batch_size))
                    predictions = self.model.predict(sections_np, batch_size=batch_size)

                    for i in range(y - q):
                        for j in range(x - q):
                            prediction = np.argmax(predictions[i * (x - q) + j], axis=2)
                            # prediction = np.reshape(prediction, (256, 256, 1))
                            prediction = prediction.astype(np.uint8)
                            prediction *= 255
                            #prediction = cv2.resize(prediction, (256, 256), cv2.INTER_CUBIC)
                            prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

                            mask[i * 256 + g * s:i * 256 + 256 + g * s, j * 256 + g * s:j * 256 + 256 + g * s] = cv2.bitwise_or(prediction, mask[i * 256 + g * s:i * 256 + 256 + g * s, j * 256 + g * s:j * 256 + 256 + g * s])
                else:
                    for i in range(len(sections)):
                        sections[i] = cv2.rotate(sections[i], cv2.ROTATE_90_CLOCKWISE)
                    # for i in range(y - q):
                    #     for j in range(x - q):
                    #         section = image[i * 256 + g * s:i * 256 + 256 + g * s, j * 256 + g * s:j * 256 + 256 + g * s]
                    #         section = cv2.rotate(section, rotM[r - 1])
                    #         section = section.astype(np.float32)
                    #         section = section / 255.0
                    #         sections.append(section)

                    sections_np = np.asarray(sections)
                    # predictions = []
                    # predictions.extend(self.model.predict(sections_np[0:int(len(sections_np)/4)], batch_size=batch_size))
                    # predictions.extend(self.model.predict(sections_np[int(len(sections_np)/4):2*int(len(sections_np) / 4)], batch_size=batch_size))
                    # predictions.extend(self.model.predict(sections_np[2*int(len(sections_np) / 4):3*int(len(sections_np) / 4)], batch_size=batch_size))
                    # predictions.extend(self.model.predict(sections_np[2 * int(len(sections_np) / 4):], batch_size=batch_size))
                    predictions = self.model.predict(sections_np, batch_size=batch_size)

                    for i in range(y - q):
                        for j in range(x - q):
                            prediction = np.argmax(predictions[i * (x - q) + j], axis=2)
                            # prediction = np.reshape(prediction, (256, 256, 1))
                            prediction = prediction.astype(np.uint8)
                            prediction *= 255
                            #prediction = cv2.resize(prediction, (256, 256), cv2.INTER_CUBIC)
                            prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
                            prediction = cv2.rotate(prediction, rotM[3 - r])

                            mask[i * 256 + g * s:i * 256 + 256 + g * s, j * 256 + g * s:j * 256 + 256 + g * s] = cv2.bitwise_or(prediction, mask[i * 256 + g * s:i * 256 + 256 + g * s, j * 256 + g * s:j * 256 + 256 + g * s])

        # split the image into 256x256 sections to feed into the FCN model
        # image pixels are also converted to floats and normalized
        # sections = []
        # for i in range(y):
        #     for j in range(x):
        #         section = image[i * 256:i * 256 + 256, j * 256:j * 256 + 256]
        #         section = section.astype(np.float32)
        #         section = section / 255.0
        #         sections.append(section)
        #
        # # needs to be numpy array
        # sections = np.asarray(sections)
        #
        # # get the FCN model to make predictions for each 256x256 section
        # predictions = self.model.predict(sections, batch_size=batch_size)
        #
        # # predictions1 = self.model.predict(sections[:int(sections.shape[0] / 4)], batch_size=batch_size)
        # # predictions2 = self.model.predict(sections[int(sections.shape[0] / 4):int(sections.shape[0] / 2)], batch_size=batch_size)
        # # predictions3 = self.model.predict(sections[int(sections.shape[0] / 2):int(sections.shape[0] / 2)+int(sections.shape[0] / 4)], batch_size=batch_size)
        # # predictions4 = self.model.predict(sections[int(sections.shape[0] / 2)+int(sections.shape[0] / 4):], batch_size=batch_size)
        # #
        # # predictions = np.concatenate((predictions1, predictions2, predictions3, predictions4), axis=0)
        #
        # # convert predictions into appropriate pixels and stitch image back together
        # for i in range(y):
        #     for j in range(x):
        #         prediction = np.argmax(predictions[i * x + j], axis=2)
        #         #prediction = np.reshape(prediction, (128, 128, 1))
        #         prediction = prediction.astype(np.uint8)
        #         prediction *= 255
        #         #prediction = cv2.resize(prediction, (256, 256), cv2.INTER_CUBIC)
        #         prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
        #
        #         image[i * 256:i * 256 + 256, j * 256:j * 256 + 256] = prediction

        # resize image back to original size (after being resized to 5cm imagery)
        mask = cv2.resize(mask, (resized_shape[1], resized_shape[0]), cv2.INTER_CUBIC)

        # convert to gray scale as color is no longer required
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        #image processing to better prep image for line segment detector
        #image = cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        mask = thinning.guo_hall_thinning(mask)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.blur(mask, (3, 3))

        return mask, resize_factor
