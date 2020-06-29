# -*- coding: utf-8 -*-


# Dataset download. It may take a while
import math
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras import datasets, layers, models
from IPython.display import clear_output, display
import tensorflow as tf
import os
import json
import sys

index = int(sys.argv[-1])

os.environ["TF_CONFIG"] = json.dumps({
    'cluster': {
        'worker': ["localhost:6666", "localhost:9999"]
    },
    'task': {'type': 'worker', 'index': index}
})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

"""### Images import
Importing the images with tensorflow. All the the reference is taken from the following links

https://www.tensorflow.org/tutorials/load_data/images
"""


AUTOTUNE = tf.data.experimental.AUTOTUNE


tf.__version__

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = pathlib.Path(dir_path+"/dataset")
data_dir.lstat()
image_count = len(list(data_dir.glob('*/*.png')))
print("Number of dataset images: " + str(image_count))

CLASS_NAMES = np.array([item.name for item in data_dir.glob(
    '*') if item.name not in ["train.txt", "validation.txt"]])
print("Labels :" + str(CLASS_NAMES))

fivers = list(data_dir.glob('5/*'))

print('Example image:\n')
for image_path in fivers[:1]:
    display(Image.open(str(image_path)))

"""### Images Loading
The following set of funtions is needed to better load the images into the database. Note that `decode img` also handles **image resizing**. <br>
It is important to set the right `IMG_HEIGHT` and `IMG_WIDTH`. Note also thath, since all image are `1280x720`, we'll be resizing them by a constant factor.
"""


# DO NOT MODIFY
ORIGINAL_HEIGHT = 720
ORIGINAL_WIDTH = 1280

# DO MODIFY
SCALE_FACTOR = 5

IMG_HEIGHT = math.floor(ORIGINAL_HEIGHT/SCALE_FACTOR)
IMG_WIDTH = math.floor(ORIGINAL_WIDTH/SCALE_FACTOR)


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img  # tf.reshape(img, [IMG_HEIGHT*IMG_WIDTH])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# split dataset (in training and validation datasets)
training_ds = np.array(['./dataset/' + item.rstrip()
                        for item in open("./dataset/train.txt", 'r')])
validation_ds = np.array(['./dataset/' + item.rstrip()
                          for item in open("./dataset/validation.txt", 'r')])
num_train = len(training_ds)
num_val = len(validation_ds)


# Creating and loading the datasets
list_training_ds = tf.data.Dataset.list_files(training_ds)
labeled_training_ds = list_training_ds.map(
    process_path, num_parallel_calls=AUTOTUNE)

list_validation_ds = tf.data.Dataset.list_files(validation_ds)
labeled_validation_ds = list_validation_ds.map(
    process_path, num_parallel_calls=AUTOTUNE)

print("Printing info for the first image. Labels are expressed in one-hot encoding")

print("\nTraining set example:")
for image, label in labeled_training_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

print("\nValidation set example:")
for image, label in labeled_validation_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

"""The following snipped shuffles and divide into batches the original database"""

BATCH_SIZE = 32


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(6, 6, n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n] == 1][0].title())
        plt.axis('off')


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)

    ds = ds.repeat()

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def prepare_for_validation(dataset, cache=True):
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()

    # all test elements in one batch
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.repeat()
    return dataset


# create a prefetch dataset for the training dataset
train_ds = prepare_for_training(labeled_training_ds)

# create a repeat dataset for the validation
validation_ds = prepare_for_validation(labeled_validation_ds)

train_ds

# try to pull one train batch and show the result
train_image_batch, train_label_batch = next(iter(train_ds))
print("First 25 images of the first training batch")
show_batch(train_image_batch.numpy(), train_label_batch.numpy())


class PlotTraining(tf.keras.callbacks.Callback):
    def __init__(self, size, sample_rate=1, zoom=1):
        self.sample_rate = sample_rate
        self.step = 0
        self.zoom = zoom
        self.steps_per_epoch = size//BATCH_SIZE

    def on_train_begin(self, logs={}):
        self.batch_history = {}
        self.batch_step = []
        self.epoch_history = {}
        self.epoch_step = []
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 7))
        plt.ioff()

    def on_batch_end(self, batch, logs={}):
        if (batch % self.sample_rate) == 0:
            self.batch_step.append(self.step)
            for k, v in logs.items():
                # do not log "batch" and "size" metrics that do not change
                # do not log training accuracy "acc"
                if k == 'batch' or k == 'size':  # or k=='acc':
                    continue
                self.batch_history.setdefault(k, []).append(v)
        self.step += 1

    def on_epoch_end(self, epoch, logs={}):
        plt.close(self.fig)
        self.axes[0].cla()
        self.axes[1].cla()

        self.axes[0].set_ylim(0, 1.2/self.zoom)
        self.axes[1].set_ylim(1-1/self.zoom/2, 1+0.1/self.zoom/2)

        self.epoch_step.append(self.step)
        for k, v in logs.items():
            # only log validation metrics
            if not k.startswith('val_'):
                continue
            self.epoch_history.setdefault(k, []).append(v)

        clear_output(wait=True)

        for k, v in self.batch_history.items():
            (self.axes[0 if k.endswith('loss') else 1]
                 .plot(np.array(self.batch_step) / self.steps_per_epoch, v, label=k))

        for k, v in self.epoch_history.items():
            (self.axes[0 if k.endswith('loss') else 1]
                 .plot(np.array(self.epoch_step) / self.steps_per_epoch, v,
                       label=k, linewidth=3))

        self.axes[0].legend()
        self.axes[1].legend()
        self.axes[0].set_xlabel('epochs')
        self.axes[1].set_xlabel('epochs')
        self.axes[0].minorticks_on()
        self.axes[0].grid(True, which='major', axis='both',
                          linestyle='-', linewidth=1)
        self.axes[0].grid(True, which='minor', axis='both',
                          linestyle=':', linewidth=0.5)
        self.axes[1].minorticks_on()
        self.axes[1].grid(True, which='major', axis='both',
                          linestyle='-', linewidth=1)
        self.axes[1].grid(True, which='minor', axis='both',
                          linestyle=':', linewidth=0.5)
        display(self.fig)


def model_1():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=5, kernel_size=5, strides=3,
                            activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    plot_training = PlotTraining(5550, sample_rate=10, zoom=5)
    history = None

    train_steps = int(num_train/BATCH_SIZE)
    validation_steps = int(num_val / BATCH_SIZE)

    EPOCHS = 20

    history = model.fit(x=train_ds,
                        validation_data=validation_ds,
                        validation_steps=validation_steps,
                        epochs=EPOCHS,
                        steps_per_epoch=train_steps,
                        callbacks=[plot_training]
                        )


def model_2():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=5, kernel_size=5, strides=3,
                            activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), padding='same'))
    model.add(layers.Conv2D(filters=8, kernel_size=5,
                            strides=2, activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=12, kernel_size=3,
                            strides=1, activation='relu'))
    model.add(layers.Conv2D(filters=15, kernel_size=3,
                            strides=1, activation='relu'))
    model.add(layers.Conv2D(filters=18, kernel_size=3,
                            strides=1, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    plot_training = PlotTraining(5550, sample_rate=10, zoom=5)
    history = None

    train_steps = int(num_train/BATCH_SIZE)
    validation_steps = int(num_val / BATCH_SIZE)

    EPOCHS = 20

    history = model.fit(x=train_ds,
                        validation_data=validation_ds,
                        validation_steps=validation_steps,
                        epochs=EPOCHS,
                        steps_per_epoch=train_steps,
                        callbacks=[plot_training]
                        )


def model_3():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=5, kernel_size=5, strides=1,
                            activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=8, kernel_size=5,
                            strides=1, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=12, kernel_size=3,
                            strides=1, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=15, kernel_size=3,
                            strides=1, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=18, kernel_size=3,
                            strides=1, activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))
    model.summary()
    plot_training = PlotTraining(5550, sample_rate=10, zoom=5)
    history = None

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_steps = int(num_train/BATCH_SIZE)
    validation_steps = int(num_val / BATCH_SIZE)

    EPOCHS = 20

    history = model.fit(x=train_ds,
                        validation_data=validation_ds,
                        validation_steps=validation_steps,
                        epochs=EPOCHS,
                        steps_per_epoch=train_steps,
                        callbacks=[plot_training]
                        )


def model_4():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=5, kernel_size=5, strides=1,
                            activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=8, kernel_size=5,
                            strides=1, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=12, kernel_size=3,
                            strides=1, activation='relu',  padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=15, kernel_size=3,
                            strides=1, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=18, kernel_size=3,
                            strides=1, activation='relu', padding='same'))
    model.add(layers.Dropout(.075))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(.075))
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    plot_training = PlotTraining(5550, sample_rate=10, zoom=5)
    history = None

    train_steps = int(num_train/BATCH_SIZE)
    validation_steps = int(num_val / BATCH_SIZE)

    EPOCHS = 20

    history = model.fit(x=train_ds,
                        validation_data=validation_ds,
                        validation_steps=validation_steps,
                        epochs=EPOCHS,
                        steps_per_epoch=train_steps,
                        callbacks=[plot_training]
                        )


def model_5():
    plot_training = PlotTraining(5550, sample_rate=10, zoom=5)
    model = models.Sequential()
    model.add(layers.Conv2D(filters=5, kernel_size=5, strides=1,
                            use_bias=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')

    model.add(layers.Conv2D(filters=8, kernel_size=5,
                            strides=1, use_bias=False, padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')

    model.add(layers.Conv2D(filters=12, kernel_size=3,
                            strides=1, use_bias=False,  padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')

    model.add(layers.Conv2D(filters=15, kernel_size=3,
                            strides=1, use_bias=False, padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')
    model.add(layers.Dropout(0.06))

    model.add(layers.Conv2D(filters=18, kernel_size=3,
                            strides=1, use_bias=False, padding='same'))

    model.add(layers.Flatten())
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')

    model.add(layers.Dense(64, use_bias=False))
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')
    model.add(layers.Dropout(0.06))

    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))
    model.summary()

    history = None
    opt = tf.keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(optimizer=opt,  # TODO: We should also tweak with the learing rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_steps = int(num_train/BATCH_SIZE)
    validation_steps = int(num_val / BATCH_SIZE)

    EPOCHS = 50  # TODO Probabilmente da aumentare

    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    history = model.fit(x=train_ds,
                        validation_data=validation_ds,
                        validation_steps=validation_steps,
                        epochs=EPOCHS,
                        steps_per_epoch=train_steps,
                        # callbacks=[plot_training, lr_decay_callback]
                        callbacks=[plot_training]
                        )


"""###Model 1"""
with strategy.scope():
    model_1()

"""###Model 2"""

# model_2()

"""###Model 3"""

# model_3()

# """###Model 4"""

# model_4()

"""###Model 5"""

# model_5()
