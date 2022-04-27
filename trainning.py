"""
+============================================================+
- Tác Giả: Hoàng Thành
- Viện Toán Ứng dụng và Tin học(SAMI - HUST)
- Email: thanh.hoangvan051199@gmail.com
- Github: https://github.com/thanhhoangvan
+============================================================+
"""

import os
# set tensorflow, keras run on CPU, Because my computer has conflict between CUDA and Tensorflow version
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory

from make_model import *


train_data = tf.keras.utils.image_dataset_from_directory('./data', image_size=(50,100))
class_names = train_data.class_names
num_classes = len(class_names)

epochs = 200
batch_size = 20
image_shape = (50,100, 3)


model = make_model(num_classes, image_shape)

history = model.fit(train_data, epochs=epochs)

model.save("my_model")
model.save("my_h5_model.h5")

acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

# predict
# img =cv2.imread('data/car/1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = np.expand_dims(img, axis=0)

# result = model.predict(img)
# class_names[np.argmax(result[0])]