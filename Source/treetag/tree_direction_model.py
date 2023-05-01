import os
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
np.seterr(all="ignore")

tf.keras.mixed_precision.set_global_policy('mixed_float16')

class TreeDirectionModel(object):

    def __init__(self, weights_path):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.applications.VGG19(include_top=False, input_shape=(128, 256, 3), classes=3,
                                                             pooling="avg", weights=None))
        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(4096, activation='relu'))
        # self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        #
        # self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        #
        # self.model.add(tf.keras.layers.Dense(32, activation='relu'))

        self.model.add(tf.keras.layers.Dense(3, activation='softmax'))

        self.model.load_weights(weights_path)

    def predict_and_interpret(self, batch, batch_images, all_lines, batch_size=256):
        predicted_lines = []
        inc_lines = []

        # feed batch into CNN to get prediction
        predictions = self.model.predict(np.asarray(batch_images), batch_size=batch_size)

        # interpret results of predictions
        # k for counter... to keep track of lines skipped
        k = 0
        # for each line in batch
        for i in range(len(batch)):
            line = batch[i]
            # if line was skipped
            if line[4] < 0:
                continue

            # get the original line
            true_line = all_lines[line[4]]

            # get the prediction for the line's direction
            j = np.argmax(predictions[k], axis=0)

            # left
            if j == 0:
                predicted_lines.append(
                    [true_line[2], true_line[3], true_line[0], true_line[1], np.amax(predictions[k], axis=0)])

            # right
            elif j == 2:
                predicted_lines.append(
                    [true_line[0], true_line[1], true_line[2], true_line[3], np.amax(predictions[k], axis=0)])

            # inconclusive
            else:
                inc_lines.append(true_line)

            # increase counter since the line wasn't skipped
            k += 1

        return predicted_lines, inc_lines
