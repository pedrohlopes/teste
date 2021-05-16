from glob import glob
import numpy as np
import keras
import datetime
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm
import h5py
from sklearn import preprocessing
from mpi4py import MPI
import json
import random
import librosa
import norbert
import pyrubberband as pyrb
import gc
from librosa.core import resample
from os import path
import os
import re
import audiofile as af
from functools import partial
physical_devices = tf.config.list_physical_devices('GPU')
# tf.compat.v1.disable_eager_execution()
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
json_path = '../norm_data_full.json'
with open(json_path) as infile:
    norm_data = json.load(infile)

# tf.debugging.set_log_device_placement(True)
dataset_path = '/home/pedro.lopes/data/audio_data/train/augmented/'
X_mean = norm_data['X_min']
X_std = norm_data['X_max'] - norm_data['X_min']


def preprocess_audio(audio, rate, return_stft=False):
    samples = []
    num_samples = 20
    offset = 1500
    freq_bins = 512
    sample_len = 128
    stft_data = tf.signal.stft(audio,
                               frame_length=1024,
                               frame_step=512,
                               fft_length=1024).numpy().T

    X_mean = norm_data['X_min']
    X_std = norm_data['X_max'] - norm_data['X_min']
    mod_input = np.zeros(
        (int(sample_len*np.ceil(len(stft_data[0])/sample_len)), 513), dtype=complex)
    mod_input[0:len(stft_data[0])] = np.array(stft_data.T)

    range_check = range(
        0, int(sample_len*np.ceil(len(stft_data[0])/sample_len)), sample_len)
    for i in range_check:
        samples.append(
            mod_input[i:i+sample_len, 1:513].T.reshape(freq_bins, sample_len, 1))
    mod_input_array = preprocess(np.array(samples), X_mean, X_std)
    if return_stft:
        return mod_input_array, stft_data, samples
    return mod_input_array


def normalize(x, save=False):
    scaled_x = (x - np.mean(x))/(np.abs(np.std(x))+1e-8)

    if save:
        return scaled_x, np.mean(x), np.std(x)
    return scaled_x


def normalize_from_outer(x, x_mean, x_std):
    scaled_x = (x - x_mean)/(x_std+1e-8)
    return scaled_x


def preprocess(sample, x_mean, x_std):
    log_sample = np.log(np.abs(sample)+1e-7)
    mod_input = normalize_from_outer(log_sample, x_mean, x_std)
    return mod_input


def preprocess_tf(sample, x_mean, x_std):
    log_sample = tf.math.log(tf.math.abs(sample)+1e-7)
    mod_input = normalize_from_outer(log_sample, x_mean, x_std)
    return mod_input


def denormalize(x, x_mean, x_std):
    scaled_x = x*(x_std + 1e-8) + x_mean
    return scaled_x


def pad_tf(stft_data):
    x_before = tf.transpose(stft_data)
    x_shape = tf.cast(tf.shape(x_before)[-1], tf.float64)
    print(x_shape)
    pad_len = tf.math.floor(tf.math.ceil(x_shape/128)*128 - x_shape)
    pad = ([0, 0], [0, pad_len])
    x = tf.pad(x_before, pad, mode='constant', constant_values=0)
    x_split = tf.split(x, 7, axis=-1)
    return x_split


def preprocess_audio_tf(x, y):
    stft_data = tf.signal.stft(x,
                               frame_length=1024,
                               frame_step=512,
                               fft_length=1024)
    x = pad_tf(stft_data)
    x = preprocess_tf(x, X_mean, X_std)[:, 0:512]
    x = tf.expand_dims(x, axis=-1)
    stft_data = tf.signal.stft(y,
                               frame_length=1024,
                               frame_step=512,
                               fft_length=1024)
    y = pad_tf(stft_data)
    y = preprocess_tf(y, X_mean, X_std)[:, 0:512]
    y = tf.expand_dims(y, axis=-1)
    return x, y


def get_unet_spleeter(input_tensor, kernel_size=(5, 5), strides=(2, 2)):
    DROPOUT = 0
    conv_activation_layer = LeakyReLU(0.2)
    deconv_activation_layer = ReLU()
    conv_n_filters = [16, 32, 64, 128, 256, 512]
    kernel_initializer = 'he_normal'
    conv2d_factory = partial(
        Conv2D, strides=strides, padding="same", kernel_initializer=kernel_initializer
    )
    # First layer.
    conv1 = conv2d_factory(conv_n_filters[0], kernel_size)(input_tensor)
    batch1 = BatchNormalization(axis=-1)(conv1)
    rel1 = conv_activation_layer(batch1)
    # Second layer.
    conv2 = conv2d_factory(conv_n_filters[1], kernel_size)(rel1)
    batch2 = BatchNormalization(axis=-1)(conv2)
    rel2 = conv_activation_layer(batch2)
    # Third layer.
    conv3 = conv2d_factory(conv_n_filters[2], kernel_size)(rel2)
    batch3 = BatchNormalization(axis=-1)(conv3)
    rel3 = conv_activation_layer(batch3)
    # Fourth layer.
    conv4 = conv2d_factory(conv_n_filters[3], kernel_size)(rel3)
    batch4 = BatchNormalization(axis=-1)(conv4)
    rel4 = conv_activation_layer(batch4)
    # Fifth layer.
    conv5 = conv2d_factory(conv_n_filters[4], kernel_size)(rel4)
    batch5 = BatchNormalization(axis=-1)(conv5)
    rel5 = conv_activation_layer(batch5)
    # Sixth layer
    conv6 = conv2d_factory(conv_n_filters[5], kernel_size)(rel5)
    batch6 = BatchNormalization(axis=-1)(conv6)
    _ = conv_activation_layer(batch6)
    #
    #
    conv2d_transpose_factory = partial(
        Conv2DTranspose,
        strides=strides,
        padding="same",
        kernel_initializer=kernel_initializer,
    )
    #
    up1 = conv2d_transpose_factory(conv_n_filters[4], kernel_size)((conv6))
    up1 = deconv_activation_layer(up1)
    batch7 = BatchNormalization(axis=-1)(up1)
    drop1 = Dropout(DROPOUT)(batch7)
    merge1 = Concatenate(axis=-1)([conv5, drop1])
    #
    up2 = conv2d_transpose_factory(conv_n_filters[3], kernel_size)((merge1))
    up2 = deconv_activation_layer(up2)
    batch8 = BatchNormalization(axis=-1)(up2)
    drop2 = Dropout(DROPOUT)(batch8)
    merge2 = Concatenate(axis=-1)([conv4, drop2])
    #
    up3 = conv2d_transpose_factory(conv_n_filters[2], kernel_size)((merge2))
    up3 = deconv_activation_layer(up3)
    batch9 = BatchNormalization(axis=-1)(up3)
    drop3 = Dropout(DROPOUT)(batch9)
    merge3 = Concatenate(axis=-1)([conv3, drop3])
    #
    up4 = conv2d_transpose_factory(conv_n_filters[1], kernel_size)((merge3))
    up4 = deconv_activation_layer(up4)
    batch10 = BatchNormalization(axis=-1)(up4)
    merge4 = Concatenate(axis=-1)([conv2, batch10])
    #
    up5 = conv2d_transpose_factory(conv_n_filters[0], kernel_size)((merge4))
    up5 = deconv_activation_layer(up5)
    batch11 = BatchNormalization(axis=-1)(up5)
    merge5 = Concatenate(axis=-1)([conv1, batch11])
    #
    up6 = conv2d_transpose_factory(1, kernel_size)((merge5))
    up6 = deconv_activation_layer(up6)
    batch12 = BatchNormalization(axis=-1)(up6)
    # Last layer to ensure initial shape reconstruction.

    up7 = Conv2D(
        1,
        (4, 4),
        dilation_rate=(2, 2),
        activation="sigmoid",
        padding="same",
        kernel_initializer=kernel_initializer,
    )((batch12))
    return up7


def custom_loss(y_true, y_pred):
    y = preprocess_audio_tf(y_true)
    return tf.keras.losses.MAE(y, y_pred)


# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# with strategy.scope():
with tf.device('GPU:0'):

    init = keras.initializers.glorot_normal(seed=None)
    reg = 5e-5
    regularizer = tf.keras.regularizers.l2(reg)
    freq_bins = 512
    sample_len = 128
    l_input = Input(shape=(512, 128, 1))
    l_out_1 = get_unet_spleeter(l_input, kernel_size=(7, 3))
    l_out_2 = get_unet_spleeter(l_input, kernel_size=(3, 7))
    concat_layer = concatenate([l_out_1, l_out_2])
    mask_layer = Conv2D(1, (1, 1))(concat_layer)
    final_layer = Multiply()([l_input, mask_layer])

#     l_input = Input(shape=(None,))
#     final_layer,mask_layer = get_unet_mask(l_input,kernel_size = (5,5))

#     l_out_2= get_unet_mask(l_input,kernel_size = (2,5))

    #l_out_1_2= get_unet_mask(l_input,kernel_size = (5,2))
    #l_out_2_2= get_unet_mask(l_input,kernel_size = (2,5))

    #concat_layer_2 = concatenate([l_out_1_2,l_out_2_2])
    #mask_layer_2 = Conv2D(1, (1, 1), activation='sigmoid')(concat_layer_2)
    #final_layer_2 = Multiply()([l_input,mask_layer_2])

    model = Model(inputs=[l_input], outputs=[final_layer])  # ,final_layer_2])
#     for layer in model.layers:
#         for attr in ['kernel_regularizer']:
#             if hasattr(layer, attr):
#                 setattr(layer, attr, regularizer)
#     #model,mask_layer = get_unet(l_input)

num_crops = 10


def make_dataset(pattern):

    names = glob(pattern)
    names_corrected = []
    for name in names:
        splitted = name.rsplit('_', maxsplit=1)[0]
        names_corrected.append(splitted)

    paths = list(dict.fromkeys(names_corrected))
    paths_tuples = [(path + '_mix.wav', path + '_vocals.wav')
                    for path in paths]
    return tf.data.Dataset.from_tensor_slices(paths_tuples)


def mapped_function(x, y):
    return preprocess_audio_tf(x), preprocess_audio_tf(y)


batch_size = 16


def load_audio(filenames):

    filenames_np = filenames.numpy()
    batch_size = filenames_np.shape[0]
    X = np.empty((batch_size, 441000))
    y = np.empty((batch_size, 441000))
    for (i, (mix_filename, vocal_filename)) in enumerate(filenames_np):
        # Reading data (line, record) from the file
        audio_mix, sr = af.read(bytes.decode(mix_filename))
        audio_vocal, sr = af.read(bytes.decode(vocal_filename))
        X[i] = audio_mix
        y[i] = audio_vocal
    return X, y


def load_audio_tf(filenames):
    (mix_filename, vocal_filename) = filenames[0], filenames[1]
    buf_mix = tf.io.read_file(
        mix_filename, name=None
    )
    audio_mix = tf.audio.decode_wav(buf_mix)[0]
    buf_vocal = tf.io.read_file(
        vocal_filename, name=None
    )
    audio_vocal = tf.audio.decode_wav(buf_vocal)[0]
    return tf.squeeze(audio_mix, axis=-1), tf.squeeze(audio_vocal, axis=-1)


def load_single_audio(filenames):
    filenames_np = filenames.numpy()
    (mix_filename, vocal_filename) = filenames_np
    audio_mix, sr = af.read(bytes.decode(mix_filename))
    audio_vocal, sr = af.read(bytes.decode(vocal_filename))
    return audio_mix, audio_vocal


def get_train_val():
    BATCH_SIZE = 128
    dataset_train = make_dataset(
        "/home/pedro.lopes/data/audio_data/train/augmented/train/*")
    dataset_val = make_dataset(
        "/home/pedro.lopes/data/audio_data/train/augmented/val/*")

    n_parallel = tf.data.experimental.AUTOTUNE
#     dataset_train = dataset_train.map(lambda x: tf.py_function(load_single_audio, [x], [tf.float64,tf.float64]),num_parallel_calls = n_parallel).batch(16).prefetch(
#                      tf.data.experimental.AUTOTUNE)
#     dataset_val = dataset_val.map(lambda x: tf.py_function(load_single_audio, [x], [tf.float64,tf.float64]),num_parallel_calls = n_parallel).batch(16).prefetch(
#                      tf.data.experimental.AUTOTUNE)

    dataset_train = dataset_train.interleave(lambda x: tf.data.Dataset.from_tensor_slices(
        [x]).map(load_audio_tf), cycle_length=n_parallel, num_parallel_calls=n_parallel)
    dataset_val = dataset_val.interleave(lambda x: tf.data.Dataset.from_tensor_slices(
        [x]).map(load_audio_tf), cycle_length=n_parallel, num_parallel_calls=n_parallel)
    dataset_train = dataset_train.map(preprocess_audio_tf, num_parallel_calls=n_parallel).unbatch().batch(BATCH_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE)
    dataset_val = dataset_val.map(preprocess_audio_tf, num_parallel_calls=n_parallel).unbatch().batch(BATCH_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE)
#     dataset_train = dataset_train.map(load_audio_tf,num_parallel_calls = n_parallel).prefetch(
#                      tf.data.experimental.AUTOTUNE)
#     dataset_val = dataset_val.map(load_audio_tf,num_parallel_calls = n_parallel).prefetch(
#                      tf.data.experimental.AUTOTUNE)

    return dataset_train, dataset_val


checkpoint_path = "../checkpoints/ws-3-7-splt-{epoch:02d}-{val_loss:.3f}_2.hdf5"
checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True,
                             monitor='loss', verbose=1, mode='auto', period=1)
log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
early_callback = EarlyStopping(monitor='loss', min_delta=1e-4, patience=10,
                               verbose=0, mode='auto', baseline=None, restore_best_weights=False)
tensorboard_callback = TensorBoard(
    log_dir=log_dir, histogram_freq=1, update_freq=15, profile_batch='200,300')

sample_rate = 44100
with tf.device('GPU:0'):
    num_epochs = 1000
    num_passes = 100
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='loss', patience=2, verbose=1, factor=0.8, min_lr=1e-5)
    params = {'dim': (freq_bins, sample_len),
              'batch_size': 8,
              'n_channels': 1,
              'shuffle': False,
              'pitch_shift': 0.3,
              'time_stretch': 0.3}
    model.compile(loss=tf.keras.losses.mae, optimizer=Adam(
        lr=3e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0))
    #model.compile(loss=tf.keras.losses.mae, optimizer= SGD(lr=1e-2, decay=1e-7, momentum=0, nesterov=True))

    # model.load_weights('../checkpoints/weights-improvement-15-0.093_2.hdf5')

    dataset_train, dataset_val = get_train_val()
    history = model.fit(x=dataset_train,
                        validation_data=dataset_val,
                        shuffle=True,
                        # validation_freq=5,
                        # workers=0,
                        use_multiprocessing=False,
                        # max_queue_size=512,
                        epochs=num_epochs,
                        callbacks=[tensorboard_callback, early_callback,
                                   checkpoint, learning_rate_reduction]
                        )
