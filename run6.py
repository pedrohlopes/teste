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
import audiofile as af
physical_devices = tf.config.list_physical_devices('GPU') 

for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
json_path = '../norm_data_full.json'
with open(json_path) as infile:
    norm_data = json.load(infile)

#tf.debugging.set_log_device_placement(True)
dataset_path = '/home/pedro.lopes/data/audio_data/train/augmented/'


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

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                shuffle=True,crop_size=30, pitch_shift=0,time_stretch = 0):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.crop_size= crop_size
        self.on_epoch_end()
        self.path = dataset_path
        self.mean_X = 0
        self.std_X = 0
        self.mean_y = 0
        self.std_y = 0
        self.pitch_shift = pitch_shift
        self.shift_range = range(-4,3,1)
        self.time_stretch= time_stretch
        self.stretch_range =  [0.5, 0.93, 1, 1.07, 1.15]
#         h5 = h5py.File(self.path, 'r')
#         self.X_train = h5.get('X_train')
#         self.Y_train_vocal = h5.get('Y_train_vocal')
#         self.Y_train_acc = h5.get('Y_train_acc')
        
        self.norm_data = norm_data

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 441000))
        y= np.empty((self.batch_size,441000))
#         y_acc= np.empty((self.batch_size,*self.dim))
        

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            #X[i,] = preprocess(self.X_train[ID:ID+128,1:513].T.reshape((*self.dim, self.n_channels)),self.norm_data['X_min'],self.norm_data['X_max'] - self.norm_data['X_min'])
            # Store class
            #y_vocal[i] = preprocess(self.Y_train_vocal[ID:ID+128,1:513].T,self.norm_data['X_min'],self.norm_data['X_max'] - self.norm_data['X_min'])
            #y_acc[i] = preprocess(self.Y_train_acc[ID:ID+128,1:513].T,self.norm_data['X_min'],self.norm_data['X_max'] - self.norm_data['X_min'])
            #sample_path,sample_half = ID.rsplit('-',maxsplit=1)
            vocal_filename = dataset_path + ID + '_vocals.wav'
            mix_filename = dataset_path + ID + '_mix.wav'
            audio_mix, sr = af.read(mix_filename)
            audio_vocal, sr = af.read(vocal_filename)
            X[i] = audio_mix[0:441000]
            y[i] = audio_vocal[0:441000]
#             for i in range(-1,2):
#                 mix_filename = dataset_path + 'sample_' + str(ID) + '_' + str(i) + '_' + 'mix.wav'
#                 audio,sr = librosa.load(mix_filename,sr=None)
#                 audios_mix.append(audio)
#                 vocal_filename = dataset_path + 'sample_' + str(ID) + '_' + str(i) + '_' + 'vocals.wav'
#                 audio,sr = librosa.load(vocal_filename,sr=None)
#                 audios_vocal.append(audio)
#             start = int(len(audio_mix)/2 * (int(sample_half)-1))
#             stop = int(len(audio_mix)/2 * int(sample_half))
             
            
             
            
#         X = np.vstack(X)
#         y = np.array(y)
#         X = np.array(X)

        return X, y#,y_acc)
    
    
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
    
    stft_data = tf.signal.stft(input_img, 
	frame_length=1024, 
	frame_step=512,
	fft_length=1024)
    x = tf.keras.backend.permute_dimensions(stft_data,(0,2,1))
    x_shape = tf.cast(tf.shape(x)[-1],tf.float64)
    pad_len = tf.math.floor(tf.math.ceil(x_shape/128)*128 - x_shape)
    pad = ([0,0],[1,0],[2,pad_len])
    x = tf.pad(x,pad,mode='constant', constant_values=0)
    X_mean = norm_data['X_min']
    X_std = norm_data['X_max'] - norm_data['X_min']
    x = tf.expand_dims(preprocess_tf(x,X_mean,X_std)[:,0:512,0:128],axis=-1)
    
        # Contracting Path 1
    c1 = conv2d_block(x, n_filters * 1, kernel_size = kernel_size, batchnorm = batchnorm,stride = 2)
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
    return c9,x
    #return mask_layer, l_out
    
def custom_loss(y_true,y_pred):
    stft_data = tf.signal.stft(y_true, 
	frame_length=1024, 
	frame_step=512,
	fft_length=1024)
    x = tf.keras.backend.permute_dimensions(stft_data,(0,2,1))
    x_shape = tf.cast(tf.shape(x)[-1],tf.float64)
    pad_len = tf.math.floor(tf.math.ceil(x_shape/128)*128 - x_shape)
    pad = ([0,0],[1,0],[2,pad_len])
    x = tf.pad(x,pad,mode='constant', constant_values=0)
    X_mean = norm_data['X_min']
    X_std = norm_data['X_max'] - norm_data['X_min']
    y = tf.expand_dims(preprocess_tf(x,X_mean,X_std)[:,0:512,0:128],axis=-1)
    return tf.keras.losses.MAE(y,y_pred)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))    
with strategy.scope():
    init = keras.initializers.glorot_normal(seed=None)
    reg = 5e-5
    regularizer = tf.keras.regularizers.l2(reg)
    freq_bins = 512
    sample_len = 128
    
    l_input = Input(shape = (None,))
    l_out_1,x= get_unet_mask(l_input,kernel_size = (7,3))
    l_out_2,_= get_unet_mask(l_input,kernel_size = (3,7))
    concat_layer = concatenate([l_out_1,l_out_2])
    mask_layer = Conv2D(1, (1, 1))(concat_layer)
    final_layer = Multiply()([x,mask_layer])

#     l_input = Input(shape=(None,))
#     final_layer,mask_layer = get_unet_mask(l_input,kernel_size = (5,5))
    
#     l_out_2= get_unet_mask(l_input,kernel_size = (2,5))
    
    #l_out_1_2= get_unet_mask(l_input,kernel_size = (5,2))
    #l_out_2_2= get_unet_mask(l_input,kernel_size = (2,5))
    
    #concat_layer_2 = concatenate([l_out_1_2,l_out_2_2])
    #mask_layer_2 = Conv2D(1, (1, 1), activation='sigmoid')(concat_layer_2)
    #final_layer_2 = Multiply()([l_input,mask_layer_2])
    
    model = Model(inputs=[l_input], outputs=[final_layer])#,final_layer_2])
    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)
    #model,mask_layer = get_unet(l_input)
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0, nesterov=True)

    adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    #model.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy'])
    #model.compile(loss=custom_loss, optimizer=adam)

from glob import glob
pattern = "/home/pedro.lopes/data/audio_data/train/augmented/train/*"
names = [path.basename(x) for x in glob(pattern)]
names_corrected =[]
for name in names:
    splitted = 'train/' + name.rsplit('_',maxsplit=1)[0]
    names_corrected.append(splitted)
training_range = list(dict.fromkeys(names_corrected))

pattern = "/home/pedro.lopes/data/audio_data/train/augmented/val/*"
names = [path.basename(x) for x in glob(pattern)]
names_corrected =[]
for name in names:
    splitted = 'val/' + name.rsplit('_',maxsplit=1)[0]
    names_corrected.append(splitted)
validation_range = list(dict.fromkeys(names_corrected))

checkpoint_path="../checkpoints/weights-improvement-{epoch:02d}-{val_loss:.2f}_2.hdf5"
checkpoint = ModelCheckpoint(checkpoint_path,save_weights_only=True, monitor='val_loss', verbose=1, mode='auto',period=10)
log_dir="../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
early_callback = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=400, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
tensorboard_callback =TensorBoard(log_dir=log_dir, histogram_freq=1,update_freq=15, profile_batch='500,520')

sample_rate = 44100
with strategy.scope():
    num_epochs = 1000
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience = 50, verbose=1,factor=0.5, min_lr=1e-5)
    #training_range= range(0,int(int(dataset_size*(1-validation_size)))) # pegando aqui os carinhas em multiplos de sample_len
    #validation_range = range(int(sample_len*int(dataset_size*(1-validation_size)/sample_len)),int(sample_len*int(dataset_size/sample_len)),sample_len) # e depois do final
    #print(training_range,validation_range)
    params = {'dim': (freq_bins,sample_len),
              'batch_size': 8,
              'n_channels': 1,
              'shuffle': False,
              'pitch_shift': 0.3,
              'time_stretch': 0.3}
    start = 5000
    #training_range = range(start,start+sample_len*num_samples,sample_len)
    #validation_generator = DataGenerator(range(64,128), **params)
    training_generator = DataGenerator(training_range, **params)
    validation_generator = DataGenerator(validation_range, **params)
    model.compile(loss=custom_loss, optimizer= Adam(lr=5e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0))
    #model.compile(loss=tf.keras.losses.mae, optimizer= SGD(lr=1e-2, decay=1e-7, momentum=0, nesterov=True))

    model.load_weights('../checkpoints/modelo_scaper2.hdf5')
    history = model.fit(x=training_generator,
                        validation_data=validation_generator,
                        workers=4,
                        use_multiprocessing=False,
                        max_queue_size=64,
                        epochs=num_epochs,
                        callbacks = [tensorboard_callback,early_callback,checkpoint,learning_rate_reduction]
                       )
