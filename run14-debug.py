import numpy as np
import keras
import datetime
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint, TensorBoard
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
import glob
physical_devices = tf.config.list_physical_devices('GPU') 
#tf.compat.v1.disable_eager_execution()
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
json_path = '../norm_data_full.json'
with open(json_path) as infile:
    norm_data = json.load(infile)

#tf.debugging.set_log_device_placement(True)
dataset_path = '/home/pedro.lopes/data/audio_data/train/augmented/'
X_mean = norm_data['X_min']
X_std = norm_data['X_max'] - norm_data['X_min']

def preprocess_audio(audio,rate,return_stft=False):
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
    mod_input = np.zeros((int(sample_len*np.ceil(len(stft_data[0])/sample_len)),513),dtype = complex)
    mod_input[0:len(stft_data[0])]= np.array(stft_data.T)

    
    range_check = range(0,int(sample_len*np.ceil(len(stft_data[0])/sample_len)),sample_len)
    for i in range_check:
        samples.append(mod_input[i:i+sample_len,1:513].T.reshape(freq_bins,sample_len,1))
    mod_input_array = preprocess(np.array(samples),X_mean,X_std)
    if return_stft:
        return mod_input_array,stft_data,samples
    return mod_input_array

def normalize(x,save=False):
    scaled_x = (x - np.mean(x))/(np.abs(np.std(x))+1e-8)

    if save:
      return scaled_x, np.mean(x), np.std(x)
    return scaled_x
def normalize_from_outer(x,x_mean,x_std):
  scaled_x = (x - x_mean)/(x_std+1e-8)
  return scaled_x
def preprocess(sample,x_mean,x_std):
  log_sample = np.log(np.abs(sample)+1e-7)
  mod_input = normalize_from_outer(log_sample,x_mean,x_std)
  return mod_input

def preprocess_tf(sample,x_mean,x_std):
  log_sample = tf.math.log(tf.math.abs(sample)+1e-7)
  mod_input = normalize_from_outer(log_sample,x_mean,x_std)
  return mod_input

def denormalize(x,x_mean,x_std):
  scaled_x = x*(x_std + 1e-8) + x_mean
  return scaled_x


def pad_tf(stft_data):
    x_before = tf.transpose(stft_data)
    x_shape = tf.cast(tf.shape(x_before)[-1],tf.float64)
    print(x_shape)
    pad_len = tf.math.floor(tf.math.ceil(x_shape/128)*128 - x_shape)
    pad = ([0,0],[0,pad_len])
    x = tf.pad(x_before,pad,mode='constant', constant_values=0)
    x_split = tf.split(x,7,axis = -1)
    return x_split

def preprocess_audio_tf(x,y):
    with tf.device('GPU:1'):
        stft_data = tf.signal.stft(x, 
        frame_length=1024, 
        frame_step=512,
        fft_length=1024)
        x = pad_tf(stft_data)
        x = preprocess_tf(x,X_mean,X_std)[:,0:512]
        x = tf.expand_dims(x,axis=-1)
        stft_data = tf.signal.stft(y, 
        frame_length=1024, 
        frame_step=512,
        fft_length=1024)
        y = pad_tf(stft_data)
        y = preprocess_tf(y,X_mean,X_std)[:,0:512]
        y = tf.expand_dims(y,axis=-1)
    return x,y




    
def conv2d_block(input_tensor, n_filters, kernel_size = (5,5),stride = 1, batchnorm = False):
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = kernel_size,\
              kernel_initializer = 'he_normal', padding = 'same', strides=(stride, stride))(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(tf.nn.leaky_relu)(x)
    
#     # second layer
#     x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
#               kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
#     if batchnorm:
#         x = BatchNormalization()(x)
#     x = Activation(tf.nn.leaky_relu)(x)
    
    return x

def conv2d_transpose_block(input_tensor,n_filters,kernel_size = (5,5),strides = (2,2), batchnorm = False):
    x = Conv2DTranspose(n_filters, kernel_size = kernel_size, strides = strides, padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)
    
    return x


def get_unet_mask(input_img, n_filters = 16, dropout = 0.5,kernel_size = (5,5), batchnorm = True):
    

        # Contracting Path 1
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = kernel_size, batchnorm = batchnorm,stride = 2)
    #c1 = MaxPooling2D((2, 2))(c1)
    #c1 = Dropout(dropout)(c1)
    
    c2 = conv2d_block(c1, n_filters * 2, kernel_size = kernel_size, batchnorm = batchnorm,stride = 2)
    #c2 = MaxPooling2D((2, 2))(c2)
    #c2 = Dropout(dropout)(c2)
    
    c3 = conv2d_block(c2, n_filters * 4, kernel_size = kernel_size, batchnorm = batchnorm,stride = 2)
    #c3 = MaxPooling2D((2, 2))(c3)
    #c3 = Dropout(dropout)(c3)
    
    c4 = conv2d_block(c3, n_filters * 8, kernel_size = kernel_size, batchnorm = batchnorm,stride = 2)
    #c4 = MaxPooling2D((2, 2))(c4)
    #c4 = Dropout(dropout)(c4)
    
    c5 = conv2d_block(c4, n_filters = n_filters * 16, kernel_size = kernel_size, batchnorm = batchnorm,stride = 2)
    #c5= MaxPooling2D((2, 2))(c5)
    #c5 = Dropout(dropout)(c5)
    
    
    c6 = conv2d_block(c5, n_filters = n_filters * 32, kernel_size = kernel_size, batchnorm = batchnorm,stride = 2)
    
    # Expansive Path 1
    u3 = conv2d_transpose_block(c6,n_filters = n_filters * 16, kernel_size = kernel_size, strides = (2, 2),batchnorm = batchnorm)
    u5 = concatenate([u3, c5])
    u5 = Dropout(dropout)(u5)
    
    u61 = conv2d_transpose_block(u5,n_filters = n_filters * 8, kernel_size = kernel_size, strides = (2, 2),batchnorm = batchnorm)
    u6 = concatenate([u61, c4])
    u6 = Dropout(dropout)(u6)
    #c6 = conv2d_block(u6, n_filters * 8, kernel_size = 5, batchnorm = batchnorm)
    
    u71 = conv2d_transpose_block(u6,n_filters = n_filters * 4, kernel_size = kernel_size, strides = (2, 2),batchnorm = batchnorm)
    u7 = concatenate([u71, c3])
    u7 = Dropout(dropout)(u7)
    #c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u81 = conv2d_transpose_block(u7,n_filters = n_filters * 2, kernel_size = kernel_size, strides = (2, 2),batchnorm = batchnorm)
    u8 = concatenate([u81, c2])
    #u8 = Dropout(dropout)(u8)
    #c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u91 = conv2d_transpose_block(u8,n_filters = n_filters * 1, kernel_size = kernel_size, strides = (2, 2),batchnorm = batchnorm)
    u9 = concatenate([u91, c1])
    #u9 = Dropout(dropout)(u9)
    #c9 = conv2d_block(u9, n_filters * 1, kernel_size = 5, batchnorm = batchnorm, stride=2)
    c9 = Conv2DTranspose(1, kernel_size = kernel_size, strides = (2, 2), padding = 'same', activation='sigmoid')(u9)
    #mask_layer = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    #l_out = Multiply()([l_input,mask_layer])
    #model = Model(inputs=[input_img], outputs=[l_out,l_out_2])
    return c9
    #return mask_layer, l_out
    
def custom_loss(y_true,y_pred):
    y=preprocess_audio_tf(y_true)
    return tf.keras.losses.MAE(y,y_pred)


# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))    
# with strategy.scope():
with tf.device('GPU:1'):

    init = keras.initializers.glorot_normal(seed=None)
    reg = 5e-5
    regularizer = tf.keras.regularizers.l2(reg)
    freq_bins = 512
    sample_len = 128
    l_input = Input(shape = (512,128,1))
    #l_out_1= get_unet_mask(l_input,kernel_size = (7,2))
    #l_out_2= get_unet_mask(l_input,kernel_size = (2,7))
    #concat_layer = concatenate([l_out_1,l_out_2])
    #mask_layer = Conv2D(1, (1, 1))(concat_layer)
    mask_layer = get_unet_mask(l_input,kernel_size = (5,5))
    final_layer = Multiply()([l_input,mask_layer])

#     l_input = Input(shape=(None,))
#     final_layer,mask_layer = get_unet_mask(l_input,kernel_size = (5,5))
    
#     l_out_2= get_unet_mask(l_input,kernel_size = (2,5))
    
    #l_out_1_2= get_unet_mask(l_input,kernel_size = (5,2))
    #l_out_2_2= get_unet_mask(l_input,kernel_size = (2,5))
    
    #concat_layer_2 = concatenate([l_out_1_2,l_out_2_2])
    #mask_layer_2 = Conv2D(1, (1, 1), activation='sigmoid')(concat_layer_2)
    #final_layer_2 = Multiply()([l_input,mask_layer_2])
    
    model = Model(inputs=[l_input], outputs=[final_layer])#,final_layer_2])
#     for layer in model.layers:
#         for attr in ['kernel_regularizer']:
#             if hasattr(layer, attr):
#                 setattr(layer, attr, regularizer)
#     #model,mask_layer = get_unet(l_input)

num_crops = 10    
from glob import glob  
def make_dataset(pattern):
    
    names = glob(pattern)
    names_corrected =[]
    for name in names:
        splitted = name.rsplit('_',maxsplit=1)[0]
        names_corrected.append(splitted)

    paths = list(dict.fromkeys(names_corrected))
    
    paths_tuples = [(path + '_mix.wav', path + '_vocals.wav') for path in paths]
    return tf.data.Dataset.from_tensor_slices(paths_tuples)

def mapped_function(x, y):
    return preprocess_audio_tf(x), preprocess_audio_tf(y)

def load_audio(filenames):
    
    filenames_np = filenames.numpy()
    batch_size = filenames_np.shape[0]
    X = np.empty((batch_size,441000))
    y = np.empty((batch_size,441000))
    for (i,(mix_filename,vocal_filename)) in enumerate(filenames_np):
        # Reading data (line, record) from the file
        audio_mix, sr = af.read(bytes.decode(mix_filename))
        audio_vocal, sr = af.read(bytes.decode(vocal_filename))
        X[i] = audio_mix
        y[i] = audio_vocal
    return X,y

def load_audio_tf(filenames):
    (mix_filename,vocal_filename) = filenames[0], filenames[1]
    buf_mix = tf.io.read_file(
        mix_filename, name=None
    )
    audio_mix = tf.audio.decode_wav(buf_mix)[0]
    buf_vocal = tf.io.read_file(
        vocal_filename, name=None
    )
    audio_vocal = tf.audio.decode_wav(buf_vocal)[0]
    return tf.squeeze(audio_mix,axis=-1),tf.squeeze(audio_vocal,axis=-1)


def load_single_audio(filenames):
    filenames_np =filenames.numpy()
    (mix_filename,vocal_filename) = filenames_np
    audio_mix, sr = af.read(bytes.decode(mix_filename))
    audio_vocal, sr = af.read(bytes.decode(vocal_filename))
    return audio_mix,audio_vocal
def get_train_val():
    BATCH_SIZE = 32
    dataset_train = make_dataset("/home/pedro.lopes/data/audio_data/train/augmented/train/*")
    dataset_val = make_dataset("/home/pedro.lopes/data/audio_data/train/augmented/train/*")
    
    n_parallel = tf.data.experimental.AUTOTUNE
#     dataset_train = dataset_train.map(lambda x: tf.py_function(load_single_audio, [x], [tf.float64,tf.float64]),num_parallel_calls = n_parallel).batch(16).prefetch(
#                      tf.data.experimental.AUTOTUNE)
#     dataset_val = dataset_val.map(lambda x: tf.py_function(load_single_audio, [x], [tf.float64,tf.float64]),num_parallel_calls = n_parallel).batch(16).prefetch(
#                      tf.data.experimental.AUTOTUNE)

    dataset_train = dataset_train.interleave(lambda x: tf.data.Dataset.from_tensor_slices([x]).map(load_audio_tf),cycle_length=n_parallel,num_parallel_calls=n_parallel)
    dataset_val = dataset_val.interleave(lambda x: tf.data.Dataset.from_tensor_slices([x]).map(load_audio_tf),cycle_length=n_parallel,num_parallel_calls=n_parallel)
    dataset_train = dataset_train.map(preprocess_audio_tf,num_parallel_calls = n_parallel).unbatch().batch(BATCH_SIZE).prefetch(
                     tf.data.experimental.AUTOTUNE)
    dataset_val = dataset_val.map(preprocess_audio_tf,num_parallel_calls = n_parallel).unbatch().batch(BATCH_SIZE).prefetch(
                     tf.data.experimental.AUTOTUNE)
#     dataset_train = dataset_train.map(load_audio_tf,num_parallel_calls = n_parallel).prefetch(
#                      tf.data.experimental.AUTOTUNE)
#     dataset_val = dataset_val.map(load_audio_tf,num_parallel_calls = n_parallel).prefetch(
#                      tf.data.experimental.AUTOTUNE)
        
    return dataset_train,dataset_val

checkpoint_path = "../checkpoints/weights-improvement-{epoch:02d}-{val_loss:.3f}_2.hdf5"
checkpoint = ModelCheckpoint(checkpoint_path,save_weights_only=True, monitor='val_loss', verbose=1, mode='auto',period=11)
log_dir="../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
early_callback = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=4, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
tensorboard_callback =TensorBoard(log_dir=log_dir, histogram_freq=1,update_freq=15, profile_batch='200,300')

sample_rate = 44100
with tf.device('GPU:1'):
    num_epochs = 1000
    num_passes = 100
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience = 5, verbose=1,factor=0.5, min_lr=1e-5, min_delta=0.001)
    params = {'dim': (freq_bins,sample_len),
              'batch_size': 8,
              'n_channels': 1,
              'shuffle': False,
              'pitch_shift': 0.3,
              'time_stretch': 0.3}
    model.compile(loss=tf.keras.losses.mae, optimizer= Adam(lr=6e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0))
    #model.compile(loss=tf.keras.losses.mae, optimizer= SGD(lr=1e-1, decay=1e-7, momentum=0, nesterov=True))
    
    model.load_weights('../checkpoints/weights-improvement-30-0.190_2.hdf5')

    dataset_train, dataset_val = get_train_val()
    history = model.fit(x=dataset_train,
                        validation_data=dataset_val,
                        shuffle=True,
                        #validation_freq=5,
                        #workers=0,
                        use_multiprocessing=False,
                        #max_queue_size=512,
                        epochs=num_epochs,
                        callbacks = [tensorboard_callback,early_callback,checkpoint,learning_rate_reduction]
                       )
