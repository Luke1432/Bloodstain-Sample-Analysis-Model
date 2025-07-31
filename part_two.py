import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

model = Sequential()

# First conv block
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Second conv block
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # ✅ Binary classification

# Compile
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'BloodstainPatternAnalysis/resized/SIZE_120_rescaled_max_area_1024/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'BloodstainPatternAnalysis/resized/SIZE_120_rescaled_max_area_1024/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


X_batch, Y_batch = next(train_generator)
print('X batch shape:', X_batch.shape)
print('Y batch:', Y_batch)

model.fit(train_generator,
            validation_data=validation_generator,
            epochs=30)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30
)

train_generator.class_indices
train_generator.classes
np.unique(train_generator.classes, return_counts=True)

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()