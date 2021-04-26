import numpy as np
import keras
import datetime
import warnings
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
from pathlib import Path
import scaper
physical_devices = tf.config.list_physical_devices('GPU') 
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
#tf.debugging.set_log_device_placement(True)
dataset_path = '/home/pedro.lopes/data/audio_data/train/'
# create foreground folder
fg_folder = Path('~/data/audio_data/train').expanduser()  
fg_folder.mkdir(parents=True, exist_ok=True)                             

# create background folder - we need to provide one even if we don't use it
bg_folder = Path('~/data/audio_data/train/mix').expanduser()
bg_folder.mkdir(parents=True, exist_ok=True)



def incoherent(fg_folder, bg_folder, event_template, seed):
    """
    This function takes the paths to the MUSDB18 source materials, an event template, 
    and a random seed, and returns an INCOHERENT mixture (audio + annotations). 
    
    Stems in INCOHERENT mixtures may come from different songs and are not temporally
    aligned.
    
    Parameters
    ----------
    fg_folder : str
        Path to the foreground source material for MUSDB18
    bg_folder : str
        Path to the background material for MUSDB18 (empty folder)
    event_template: dict
        Dictionary containing a template of probabilistic event parameters
    seed : int or np.random.RandomState()
        Seed for setting the Scaper object's random state. Different seeds will 
        generate different mixtures for the same source material and event template.
        
    Returns
    -------
    mixture_audio : np.ndarray
        Audio signal for the mixture
    mixture_jams : np.ndarray
        JAMS annotation for the mixture
    annotation_list : list
        Simple annotation in list format
    stem_audio_list : list
        List containing the audio signals of the stems that comprise the mixture
    """
    
    # Create scaper object and seed random state
    sc = scaper.Scaper(
        duration=10.0,
        fg_path=str(fg_folder),
        bg_path=str(bg_folder),
        random_state=seed
    )
    
    # Set sample rate, reference dB, and channels (mono)
    sc.sr = 44100
    sc.ref_db = -20
    sc.n_channels = 1
    
    # Copy the template so we can change it
    event_parameters = event_template.copy()
    
    # Iterate over stem types and add INCOHERENT events
    labels = ['vocals', 'acc']
    for label in labels:
        event_parameters['label'] = ('const', label)
        sc.add_event(**event_parameters)
    
    # Return the generated mixture audio + annotations 
    # while ensuring we prevent audio clipping
    return sc.generate(disable_sox_warnings=True,fix_clipping=True)


def coherent(fg_folder, bg_folder, event_template, seed):
    """
    This function takes the paths to the MUSDB18 source materials and a random seed,
    and returns an COHERENT mixture (audio + annotations).
    
    Stems in COHERENT mixtures come from the same song and are temporally aligned.
    
    Parameters
    ----------
    fg_folder : str
        Path to the foreground source material for MUSDB18
    bg_folder : str
        Path to the background material for MUSDB18 (empty folder)
    event_template: dict
        Dictionary containing a template of probabilistic event parameters
    seed : int or np.random.RandomState()
        Seed for setting the Scaper object's random state. Different seeds will 
        generate different mixtures for the same source material and event template.
        
    Returns
    -------
    mixture_audio : np.ndarray
        Audio signal for the mixture
    mixture_jams : np.ndarray
        JAMS annotation for the mixture
    annotation_list : list
        Simple annotation in list format
    stem_audio_list : list
        List containing the audio signals of the stems that comprise the mixture
    """
        
    # Create scaper object and seed random state
    sc = scaper.Scaper(
        duration=10.0,
        fg_path=str(fg_folder),
        bg_path=str(bg_folder),
        random_state=seed
    )
    
    # Set sample rate, reference dB, and channels (mono)
    sc.sr = 44100
    sc.ref_db = -20
    sc.n_channels = 1
    
    # Copy the template so we can change it
    event_parameters = event_template.copy()    
    
    # Instatiate the template once to randomly choose a song,   
    # a start time for the sources, a pitch shift and a time    
    # stretch. These values must remain COHERENT across all stems
    sc.add_event(**event_parameters)
    event = sc._instantiate_event(sc.fg_spec[0])
    
    # Reset the Scaper object's the event specification
    sc.reset_fg_event_spec()
    
    # Replace the distributions for source time, pitch shift and 
    # time stretch with the constant values we just sampled, to  
    # ensure our added events (stems) are coherent.              
    event_parameters['source_time'] = ('const', event.source_time)
    event_parameters['pitch_shift'] = ('const', event.pitch_shift)
    event_parameters['time_stretch'] = ('const', event.time_stretch)

    # Iterate over the four stems (vocals, drums, bass, other) and 
    # add COHERENT events.                                         
    labels = ['vocals', 'acc']
    for label in labels:
        
        # Set the label to the stem we are adding
        event_parameters['label'] = ('const', label)
        
        # To ensure coherent source files (all from the same song), we leverage
        # the fact that all the stems from the same song have the same filename.
        # All we have to do is replace the stem file's parent folder name from "vocals" 
        # to the label we are adding in this iteration of the loop, which will give the 
        # correct path to the stem source file for this current label.
        coherent_source_file = event.source_file.replace('vocals', label)
        event_parameters['source_file'] = ('const', coherent_source_file)
        # Add the event using the modified, COHERENT, event parameters
        sc.add_event(**event_parameters)
    
    # Generate and return the mixture audio, stem audio, and annotations
    return sc.generate(fix_clipping=True,disable_sox_warnings=True)

import nussl

def generate_mixture(dataset, fg_folder, bg_folder, event_template, seed):
    coherent_prob = .5
    # hide warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        # flip a coint to choose coherent or incoherent mixing
        random_state = np.random.RandomState(seed)
        
        # generate mixture
        if random_state.rand() < coherent_prob:
            data = coherent(fg_folder, bg_folder, event_template, seed)
        else:
            data = incoherent(fg_folder, bg_folder, event_template, seed)
            
    # unpack the data
    mixture_audio, mixture_jam, annotation_list, stem_audio_list = data
    
    # convert mixture to nussl format
    mix = dataset._load_audio_from_array(
        audio_data=mixture_audio, sample_rate=dataset.sample_rate
    )
    
    # convert stems to nussl format
    sources = {}
    ann = mixture_jam.annotations.search(namespace='scaper')[0]
    for obs, stem_audio in zip(ann.data, stem_audio_list):
        key = obs.value['label']
        sources[key] = dataset._load_audio_from_array(
            audio_data=stem_audio, sample_rate=dataset.sample_rate
        )
    
    # store the mixture, stems and JAMS annotation in the format expected by nussl
    output = {
        'mix': mix,
        'sources': sources,
        'metadata': mixture_jam
    }
    return output

class MixClosure:
    
    def __init__(self, fg_folder, bg_folder, event_template):
        self.fg_folder = fg_folder
        self.bg_folder = bg_folder
        self.event_template = event_template
        
    def __call__(self, dataset, seed):
        return generate_mixture(dataset, self.fg_folder, self.bg_folder, self.event_template, seed)
    
# Initialize our mixing function with our specific source material and event template
template_event_parameters = {
                'label': ('const', 'vocals'),
                'source_file': ('choose', []),
                'source_time': ('uniform', 0, 600),
                'event_time': ('const', 0),
                'event_duration': ('const', 10.0),
                'snr': ('uniform', -5, 5),
                'pitch_shift': ('uniform', -2, 2),
                'time_stretch': ('uniform', 0.8, 1.2)
            }
mix_func = MixClosure(fg_folder, bg_folder, template_event_parameters)

# Create a nussle OnTheFly data generator
on_the_fly = nussl.datasets.OnTheFly(
    num_mixtures=5000,
    mix_closure=mix_func
)


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
    
    json_path = '../norm_data_full.json'
    with open(json_path) as infile:
        norm_data = json.load(infile)
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
def denormalize(x,x_mean,x_std):
  scaled_x = x*(x_std + 1e-8) + x_mean
  return scaled_x

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                shuffle=True,crop_size=30, pitch_shift=0,time_stretch = 0, sample_rate = 44100):
        'Initialization'
        self.dim = dim
        self.sr = sample_rate
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
        norm_path = '../norm_data_full.json'
        with open(norm_path, mode ='r') as fp:
            self.norm_data = json.load(fp)

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
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y_vocal= np.empty((self.batch_size,*self.dim))
#         y_acc= np.empty((self.batch_size,*self.dim))
        
        
        X = []
        y = []
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            item = on_the_fly[ID]
            mix = item['mix'].audio_data
            vocals = item['sources']['vocals'].audio_data
                
                
            
            X.append(preprocess_audio(mix,self.sr))
            y.append(preprocess_audio(vocals,self.sr))
#             for i in range(-1,2):
#                 mix_filename = dataset_path + 'sample_' + str(ID) + '_' + str(i) + '_' + 'mix.wav'
#                 audio,sr = librosa.load(mix_filename,sr=None)
#                 audios_mix.append(audio)
#                 vocal_filename = dataset_path + 'sample_' + str(ID) + '_' + str(i) + '_' + 'vocals.wav'
#                 audio,sr = librosa.load(vocal_filename,sr=None)
#                 audios_vocal.append(audio)
#             start = int(len(audio_mix)/2 * (int(sample_half)-1))
#             stop = int(len(audio_mix)/2 * int(sample_half))
             
            
             
            
        X = np.vstack(X)
        y = np.vstack(y)
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
    
with tf.device('/device:GPU:0'):
    init = keras.initializers.glorot_normal(seed=None)
    reg = 5e-5
    regularizer = tf.keras.regularizers.l2(reg)
    freq_bins = 512
    sample_len = 128
    
    l_input = Input(shape = (freq_bins,sample_len,1))
    l_out_1= get_unet_mask(l_input,kernel_size = (7,3))
    l_out_2= get_unet_mask(l_input,kernel_size = (3,7))
    concat_layer = concatenate([l_out_1,l_out_2])
    mask_layer = Conv2D(1, (1, 1))(concat_layer)
    final_layer = Multiply()([l_input,mask_layer])

#     l_input = Input(shape = (freq_bins,sample_len,1))
#     mask_layer = get_unet_mask(l_input,kernel_size = (5,5))
#     final_layer = Multiply()([l_input,mask_layer])
    
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

json_path = '../len_full_samples.json'
with open(json_path) as infile:
    len_json = json.load(infile)
checkpoint_path="../checkpoints/weights-improvement-{epoch:02d}-{val_loss:.2f}_2.hdf5"
checkpoint = ModelCheckpoint(checkpoint_path,save_weights_only=True, monitor='val_loss', verbose=1, mode='auto',period=10)
log_dir="../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
early_callback = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=400, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
validation_range = range(0,150)
training_range = range(150,1000)


sample_rate = 44100
with tf.device('/device:GPU:0'):
    num_epochs = 1000
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience = 50, verbose=1,factor=0.5, min_lr=1e-5)
    #training_range= range(0,int(int(dataset_size*(1-validation_size)))) # pegando aqui os carinhas em multiplos de sample_len
    #validation_range = range(int(sample_len*int(dataset_size*(1-validation_size)/sample_len)),int(sample_len*int(dataset_size/sample_len)),sample_len) # e depois do final
    #print(training_range,validation_range)
    params = {'dim': (freq_bins,sample_len),
              'batch_size': 8,
              'n_channels': 1,
              'shuffle': True,
              'pitch_shift': 0.3,
              'time_stretch': 0.3}
    start = 5000
    #training_range = range(start,start+sample_len*num_samples,sample_len)
    #validation_generator = DataGenerator(range(64,128), **params)
    training_generator = DataGenerator(training_range, **params)
    validation_generator = DataGenerator(validation_range, **params)
    model.compile(loss=tf.keras.losses.mae, optimizer= Adam(lr=5e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0))
    #model.compile(loss=tf.keras.losses.mae, optimizer= SGD(lr=1e-2, decay=1e-7, momentum=0, nesterov=True))

    tensorboard_callback =TensorBoard(log_dir=log_dir, histogram_freq=1,update_freq=15)
    model.load_weights('../checkpoints/melhor_ate_agora2.hdf5')
    history = model.fit(x=training_generator,
                        validation_data=validation_generator,
                        workers=1,
                        use_multiprocessing=False,
                        max_queue_size=256,
                        epochs=num_epochs,
                        callbacks = [tensorboard_callback,early_callback,checkpoint,learning_rate_reduction]
                       )