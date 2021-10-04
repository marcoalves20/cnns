import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class Cnn_architecture():
    def __init__(self, architecture, input_shape, nclasses):
        self.input_shape = input_shape
        self.nclasses = nclasses
        self.arch = architecture

    def create(self):#
        print('initializing Cnn architecture: ' + self.arch)
        if self.arch == 'VGG16':
            self.backbone = VGG16(weights="imagenet", include_top=False,
                                  input_tensor=Input(
                                      shape=(self.input_shape[0], self.input_shape[1], 3)))
        elif self.arch == 'ResNet50':
            self.backbone = ResNet50(weights="imagenet", include_top=False,
                                     input_tensor=Input(
                                         shape=(self.input_shape[0], self.input_shape[1], 3)))
        elif self.arch == 'InceptionV3':
            self.backbone = InceptionV3(weights="imagenet", include_top=False,
                                     input_tensor=Input(
                                         shape=(self.input_shape[0], self.input_shape[1], 3)))
        elif self.arch == 'MobileNetV2':
            self.backbone = MobileNetV2(weights="imagenet", include_top=False,
                                     input_tensor=Input(
                                         shape=(self.input_shape[0], self.input_shape[1], 3)))
        else:
            raise Exception('selected cnn architecture not supported')

        self.backbone.trainable = False
        flatten = self.backbone.output
        flatten = Flatten()(flatten)

        # fully connected layer for the bbox output branch
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

        # fully connected layer for the class prediction branch
        softmaxHead = Dense(512, activation="relu")(flatten)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(512, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(self.nclasses, activation="softmax", name="class_label")(softmaxHead)

        # assemble final model
        self.model = Model(inputs=self.backbone.input, outputs=(bboxHead, softmaxHead))

