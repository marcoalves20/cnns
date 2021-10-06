import numpy as np
from tensorflow.keras.models import Sequential
import cv2
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


class DataGenerator(Sequential):
    def __init__(self, df, batch_size, input_size, nclasses, shuffle=True ):
        self.df = df
        self.n_classes = nclasses
        self.labels = self.df.iloc[:,-nclasses:].to_numpy()
        self.df = self.df.drop(df.columns[-self.n_classes:], axis=1)
        self.list_IDs = np.arange(len(self.df))
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.on_epoch_end()
        self.count = 0

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        # shuffle the indexes of the training data after the end of the epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # return number of batches
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        if self.count >= self.__len__():
            self.count = 0
            self.on_epoch_end()

        indexes = self.indexes[self.count*self.batch_size: (self.count + 1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        self.count += 1
        # return data of batch of index = index
        return self.generate_data(list_IDs_temp)

    def generate_data(self, list_IDs_temp):
        # Generates data for a single batch
        # initialize X and Y
        X = np.empty((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]))
        Y1 = np.empty((self.batch_size, 4))
        Y2 = np.empty((self.batch_size, self.n_classes))

        # get data
        for i, id in enumerate(list_IDs_temp):

            filename, startY, endY, startX, endX = self.df.iloc[id,:]
            image = cv2.imread(filename)
            h, w = image.shape[:2]
            # scale bounding boxes based on image resolution
            startX = float(startX) / w
            endX = float(endX) / w
            startY = float(startY) / h
            endY = float(endY) / h
            image = load_img(filename, target_size=(self.input_size[0], self.input_size[1]))
            image = img_to_array(image)
            # store image in X vector
            X[i,] = image
            # store bounding box parameters in Y1 vector
            Y1[i,] = np.array([startX, startY, endX, endY])
            # store class label in Y2 vector
            Y2[i,] = self.labels[id,:]

        # Normalize images
        X = X / 255.0
        return  X, [Y1,Y2]


if __name__ == '__main__':
    data_path = 'dataset/data.csv'
    df = pd.read_csv(data_path)

    # label encoding at the dataframe level
    labels = df.iloc[:, -1]
    label_enc = LabelBinarizer()
    labels = label_enc.fit_transform(labels)

    # combine encoded labels with dataframe
    df = df.drop(df.columns[-1], axis=1)
    labels_df = pd.DataFrame(labels)
    df = pd.concat([df, labels_df], axis=1)

    train_dataset = DataGenerator(df=df, batch_size=8, input_size=[224,224,3], nclasses=6)
    # testing dataloader

    i = 0
    while i < train_dataset.__len__() :
        X, Y = next(iter(train_dataset))
        print(X,Y)



