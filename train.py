# from pyimagesearch import config
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from dataloader import DataGenerator
from models import Cnn_architecture
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

myModels = ['VGG16', 'ResNet50', 'InceptionV3', 'MobileNetV2']
for chosenModel in myModels:
    data_path = 'dataset/data.csv'
    df = pd.read_csv(data_path)

    # label encoding at the dataframe level
    labels = df.iloc[:,-1]
    label_enc = LabelBinarizer()
    labels = label_enc.fit_transform(labels)

    # combine encoded labels with dataframe
    df = df.drop(df.columns[-1], axis=1)
    labels_df = pd.DataFrame(labels)
    df = pd.concat([df,labels_df], axis=1)

    # split data for training and validation
    split = train_test_split(df, test_size=0.3, shuffle=True)
    nclasses = labels.shape[-1]
    # initialize data generator for training and validation
    train_generator = DataGenerator(df=split[0], batch_size=8, input_size=[224,224,3], nclasses = nclasses)
    validation_generator = DataGenerator(df=split[1], batch_size=16, input_size=[224,224,3], nclasses = nclasses)

    # ================ Model training ===================
    Cnn = Cnn_architecture(architecture=chosenModel, input_shape=[224,224], nclasses=nclasses)
    Cnn.create()

    # loss functions and optimizer
    losses = {
        "class_label": "categorical_crossentropy",
        "bounding_box": "mean_squared_error"}

    lossWeights = {
        "class_label": 1.0,
        "bounding_box": 1.0
    }



    def batch_generator(train_generator):
        while True:
            X, Y = next(iter(train_generator))
            yield(X, {'bounding_box': Y[0], 'class_label': Y[1]} )

    # Train model
    opt = Adam(lr=1e-5)
    Cnn.model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
    H = Cnn.model.fit(batch_generator(train_generator), validation_data=batch_generator(validation_generator),
                            epochs=3, steps_per_epoch=train_generator.__len__(), verbose=1,
                            validation_steps=validation_generator.__len__())

    #set all layers to trainable
    Cnn.model.trainable = True
    opt = Adam(lr=1e-6)
    Cnn.model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
    H = Cnn.model.fit(batch_generator(train_generator), validation_data=batch_generator(validation_generator),
                            epochs=15, steps_per_epoch=train_generator.__len__(), verbose=1,
                            validation_steps=validation_generator.__len__())

    Cnn.model.save('output/'+Cnn.arch+'_detector.h5', save_format='h5')
    f = open('output/labelEncoder.pkl', "wb")
    f.write(pickle.dumps(label_enc))
    f.close()