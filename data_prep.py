import scipy.io
import numpy as np
import os
import csv
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

categories = os.listdir('dataset/annotations/')
annotations = []
for cat in categories:
    files = sorted(os.listdir('dataset/annotations/' + cat + '/'))
    img_path = 'dataset/images/' + cat
    for count, anotation in enumerate(files):
        print(count)
        mat = scipy.io.loadmat('dataset/annotations/' + cat + '/' + anotation)
        temp = [img_path + '/image_'+'%04d' % (count+1) + '.jpg'] + np.squeeze(mat['box_coord']).tolist() + [cat]
        annotations.append(temp)
df = pd.DataFrame(np.asarray(annotations))
df.to_csv('dataset/data.csv', index=False)
