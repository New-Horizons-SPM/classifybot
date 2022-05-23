#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:30:07 2022

https://www.tensorflow.org/tutorials/load_data/images

@author: jack
"""

from datetime import datetime
import pickle
# import numpy as np
import os
# import PIL
# import PIL.Image
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
# import tensorflow_datasets as tfds

import zulip


# import matplotlib
# matplotlib.use('Agg') ## for plotting headless
# import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import pathlib

data_dir = pathlib.Path('image_data')

# if not pickle.load(open('retrain_flag.pkl', 'rb')):
#     return

master_label_dict = {}
for root, dirs, files in os.walk(data_dir):
    for name in files:
        if name == 'file_labels.pkl':
            master_label_dict.update(pickle.load(open(os.path.join(root, name), 'rb')))

## make ordered lists of the dict keys and values
data_files = []
data_labels = []
for key, value in master_label_dict.items():
    data_files.append(os.path.join(data_dir, key))
    data_labels.append(value)
    

            
X_train, X_val, y_train, y_val = train_test_split(data_files, data_labels, test_size=0.2, random_state=44)


## from https://github.com/ashrefm/multi-label-soft-f1/blob/master/Multi-Label%20Image%20Classification%20in%20TensorFlow%202.0.ipynb
mlb = MultiLabelBinarizer()
mlb.fit(y_train)
N_LABELS = len(mlb.classes_)

# # Loop over all labels and show them
# N_LABELS = len(mlb.classes_)
# for (i, label) in enumerate(mlb.classes_):
#     print("{}. {}".format(i, label))

y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)


IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
CHANNELS = 3 # Keep RGB color channels to match the input format of the model

def parse_function(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label

BATCH_SIZE = 256 # Big enough to measure an F1-score
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 25 # Shuffle the training data by a chunk of 1024 observations

def create_dataset(filenames, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    
    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset


train_ds = create_dataset(X_train, y_train_bin)
val_ds = create_dataset(X_val, y_val_bin)


### headless model

model = tf.keras.Sequential()

model.add(hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5",
                   trainable=False))

model.add(layers.Dense(256, activation='relu', name='hidden_layer_1'))
model.add(layers.Dense(256, activation='relu', name='hidden_layer_2'))
model.add(layers.Dense(N_LABELS, activation='sigmoid', name='output'))

model.build([None, IMG_SIZE, IMG_SIZE, CHANNELS])

model.summary()

from tf_custom_metric import macro_soft_f1

LR = 1e-5 # keep it small when transfer learning
EPOCHS = 30

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
  loss=macro_soft_f1,
  metrics=[macro_soft_f1])

start = datetime.now()
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=create_dataset(X_val, y_val_bin))
print('\nTraining took {}'.format(datetime.now()-start))



model.save('kf_model.model')

with open('class_names.pkl', 'wb') as f:
    pickle.dump(mlb.classes_, f)
    
print(mlb.classes_)

## zulip message saying the training is done
client = zulip.Client(config_file='zuliprc')

request = {
    "type": "stream",
    "to": "scanbot",
    "topic": "TF model",
    "content": "train_model finished"
    }
result = client.send_message(request)
print(result)
