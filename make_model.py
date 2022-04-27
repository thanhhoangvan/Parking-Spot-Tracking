import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Rescaling
from tensorflow.keras.utils import image_dataset_from_directory

def make_model(num_classes=2, img_shape=(50, 100, 3)):

    data_augmentation = Sequential(
        [
            RandomFlip("horizontal", input_shape=img_shape),
            RandomRotation(0.1),
            RandomZoom(0.1),
        ]
    )

    model = Sequential([
    data_augmentation,
    Rescaling(1./255),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    print(model.summary())
    return model
