import scaper
import random
import audiofile as af
from glob import glob
from os import path
import numpy as np

import json

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
    return sc.generate(fix_clipping=False)


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
    return sc.generate(fix_clipping=False)