import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    directory='resized\\SIZE_120_rescaled_max_area_1024\\train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    directory='resized\\SIZE_120_rescaled_max_area_1024\\test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

X_train, Y_train = next(train_generator)
print('X train:', np.array(X_train))
print('Y train:', Y_train)
