# from pyimagesearch import config
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

config = {'annotations_path': 'dataset/annotations/',
          'img_path': 'dataset/images/',
          'output_path': 'output/',
          'input_shape': [224,224],
          'batch_size': 16,
          'n_epochs': 10}

print('loading dataset')
data = []
labels = []
bboxes = []
img_paths = []

for csvPath in paths.list_files(config['annotations_path'], validExts=".csv"):
    rows = open(csvPath).read().strip().split("\n")
    for row in rows:
        row = row.split(",")
        filename, startY, endY, startX, endX, label = row
        imagePath = config['img_path'] + label + '/' + filename
        image = cv2.imread(imagePath)
        h, w = image.shape[:2]
        # scale bounding boxes based on image resolution
        startX = float(startX) / w
        endX = float(endX) / w
        startY = float(startY) / h
        endY = float(endY) / h
        image = load_img(imagePath, target_size=(config['input_shape'][0], config['input_shape'][1]))
        image = img_to_array(image)

        data.append(image)
        labels.append(label)
        bboxes.append([startX, startY, endX, endY])
        img_paths.append(imagePath)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes)
img_paths = np.array(img_paths)

# label encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# split data is train and test
split = train_test_split(data,labels,bboxes,img_paths, test_size=0.2, random_state=42)
trainImages, testImages = split[:2]
trainLabels, testLabels = split[2:4]
trainBboxes, testBboxes = split[4:6]
trainPaths, testPaths = split[6:]



# ================ Model training ===================
vgg = VGG16(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(config['input_shape'][0],config['input_shape'][1],3)))
vgg.trainable = False

flatten = vgg.output
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
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

# assemble final model
model = Model(inputs=vgg.input, outputs=(bboxHead, softmaxHead))

# loss functions and optimizer
losses = {
	"class_label": "categorical_crossentropy",
	"bounding_box": "mean_squared_error"}

lossWeights = {
	"class_label": 1.0,
	"bounding_box": 1.0
}

opt = Adam(lr=1e-4)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())


# dictionaries for target training and testing outputs
trainTargets = {"class_label": trainLabels, "bounding_box": trainBboxes}
testTargets = {"class_label": testLabels, "bounding_box": testBboxes}

# Train model
H = model.fit(trainImages, trainTargets,
              validation_data=(testImages, testTargets),
              batch_size=config['batch_size'],epochs=config['n_epochs'], verbose=1)

model.save(config['output_path']+'detector.h5', save_format='h5')
f = open(config['output_path']+'lb.pkl', "wb")
f.write(pickle.dumps(lb))
f.close()


lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, 5)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
# loop over the loss names
for (i, l) in enumerate(lossNames):
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history["val_" + l], label="val_" + l)
    ax[i].legend()
plt.show()
