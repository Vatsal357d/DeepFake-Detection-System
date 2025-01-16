import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

train_dir = r'C:\Users\vatsa\OneDrive\Desktop\programs\python files\deepfake\deepfake_dataset\train'
test_dir = r'C:\Users\vatsa\OneDrive\Desktop\programs\python files\deepfake\deepfake_dataset\test'

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=42,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=1, 
    class_mode='binary'
)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) 

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) 

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_gen, epochs=10, validation_data=test_gen)

model.save('deepfake_detector_model.h5')

test_loss, test_acc = model.evaluate(test_gen)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

y_pred = model.predict(test_gen)
y_pred = (y_pred > 0.5).astype(int)

print("Classification Report:")
print(classification_report(test_gen.classes, y_pred, target_names=['Real', 'Fake']))

print("Confusion Matrix:")
print(confusion_matrix(test_gen.classes, y_pred))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
