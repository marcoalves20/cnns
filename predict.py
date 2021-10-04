# import the necessary packages
# from pyimagesearch import config
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
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

model = load_model('output/VGG16_detector.h5')
label_enc = pickle.loads(open('output/labelEncoder.pkl', "rb").read())

categories = os.listdir('dataset/images/')
categories.pop(2)
images_dir = []

for image_dir in os.listdir('dataset/testing/'):
    image_dir = 'dataset/testing/' + image_dir
    image = load_img(image_dir, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]
    i = np.argmax(labelPreds, axis=1)
    label = label_enc.classes_[i][0]

    image = cv2.imread(image_dir)
    (h, w) = image.shape[:2]
    # scale the predicted bounding box coordinates based on the image
    # dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    # draw the predicted bounding box and class label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (startX, startY), (endX, endY),
                  (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()