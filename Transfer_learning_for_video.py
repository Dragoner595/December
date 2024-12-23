import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ufc_101_subset(URL, 
                        num_classes = 10, 
                        splits = {"train": 30, "test": 20}, 
                        download_dir = download_dir)

batch_size = 8
num_frames = 8

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'],num_frames,training = True),
                                          output_signature = output_signature)

train_ds = train_ds.batch(batch_size)

for frames, labels in train_ds.take(10):
  print(labels)

print(f'Shape:{frames.shape}')
print(f'Label:{labels.shape}')

gru = layers.GRU(units= 4 , return_sequences = True , return_state = True )

inputs = tf.random.normal(shape = [1,10,8]) # (batch , seqence ,channels)

result, state = gru(inputs) # Run it all at once

first_half, state = gru(inputs[:,:5,:]) # run the first half , and capture the state
second_half, _ = gru (inputs[:,5:,:],initial_state = state)  # USa the state to continue where you left off

print(np.allclose(result[:,:5,:],first_half))
print(np.allclose(result[:,5:,:], second_half))