import os 
import numpy as np
import librosa
import librosa.display
from pyparsing import col
from scipy import signal

import utils

import event_manager

SAMPLING_RATE_DEFAULT = 4000

FEATURE_NAME = ['rmse', 'chroma_stft1', 'chroma_stft2', 'chroma_stft3', 'chroma_stft4',
       'chroma_stft5', 'chroma_stft6', 'chroma_stft7', 'chroma_stft8',
       'chroma_stft9', 'chroma_stft10', 'chroma_stft11', 'chroma_stft12',
       'spec_cent', 'spec_bw', 'rolloff', 'zcr', 'mfcc1', 'mfcc2', 'mfcc3',
       'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
       'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18',
       'mfcc19','mfcc20', 'delta_mfcc1', 'delta_mfcc2', 'delta_mfcc3', 'delta_mfcc4',
       'delta_mfcc5', 'delta_mfcc6', 'delta_mfcc7', 'delta_mfcc8',
       'delta_mfcc9', 'delta_mfcc10', 'delta_mfcc11', 'delta_mfcc12',
       'delta_mfcc13', 'delta_mfcc14', 'delta_mfcc15', 'delta_mfcc16',
       'delta_mfcc17', 'delta_mfcc18', 'delta_mfcc19', 'delta_mfcc20',
       'delta2_mfcc1', 'delta2_mfcc2', 'delta2_mfcc3', 'delta2_mfcc4',
       'delta2_mfcc5', 'delta2_mfcc6', 'delta2_mfcc7', 'delta2_mfcc8',
       'delta2_mfcc9', 'delta2_mfcc10', 'delta2_mfcc11', 'delta2_mfcc12',
       'delta2_mfcc13', 'delta2_mfcc14', 'delta2_mfcc15', 'delta2_mfcc16',
       'delta2_mfcc17', 'delta2_mfcc18', 'delta2_mfcc19', 'delta2_mfcc20']

# extract data and info
def get_list_recording(audio_folder):
    list_files = []
    for file in os.listdir(audio_folder):
        if file.endswith('.wav'):
            audio_file = file.split('.wav')[0]
            list_files.append(audio_file)
    return list_files

def load_file_from_recording_name(audio_folder,audio_subject):
    audio_data_file = f'{audio_subject}.wav'
    audio_text_file = f'{audio_subject}.txt'
    audio_event_file = f'{audio_subject}_events.txt'
    audio_data_file = os.path.join(audio_folder,audio_data_file)
    audio_text_file = os.path.join(audio_folder,audio_text_file)
    audio_event_file = os.path.join(audio_folder,audio_event_file)
    return {'data':audio_data_file,
            'event':audio_event_file,
            'text':audio_text_file}
    
def extract_data_from_file(file,fs= SAMPLING_RATE_DEFAULT):
    x_data, sr = librosa.load(file, sr=fs)
    audio_data = {
        'data': x_data,
        'time': np.linspace(0,len(x_data)/sr,len(x_data))
        }
    audio_data['fs'] = sr
    return audio_data

#Processing
def apply_filtering_on_signal(audio_data, lf=None, hf= None, backward = True):
    if lf and hf:
        b, a = signal.butter(4, [lf, hf], btype='band',fs=audio_data['fs'])
    elif lf and not hf: 
        b, a = signal.butter(4, lf, btype='low',fs=audio_data['fs'])
    elif hf and not lf: 
        b, a = signal.butter(4, hf, btype='high', fs=audio_data['fs'])
    
    if backward:
        data_filtered = signal.filtfilt(b, a, audio_data['data'])
    else:
        data_filtered = signal.filter(b, a, audio_data['data'])

    audio_data['data'] = data_filtered 
    return audio_data


def apply_stft(data,n_fft=2048,win_shift=None,center=False):
    if not win_shift:
        win_shift = n_fft/4
    X = librosa.stft(data['data'],n_fft=n_fft, center=center)
    Xdb = librosa.amplitude_to_db(abs(X))
    times = librosa.times_like(Xdb,sr=data['fs'],n_fft=n_fft,hop_length=win_shift)
    freqs = np.arange(0, 1 + n_fft / 2) * data['fs'] / n_fft
    return {
        'data': Xdb,
        'time': times,
        'freq':freqs,
        'fs':data['fs']
    }
def apply_wavelets(data):
    cs1, f1 = utils.cwt2(data['data'], nv=12, sr=data['fs'],low_freq=60)
    data_wv = data.copy()
    data_wv['data'] = np.log10(np.abs(cs1))
    data_wv['freq'] = f1[::-1]
    return data_wv

def preprocess_data(audio_dict, annotations='text'):
    data = extract_data_from_file(audio_dict['data'])
    if annotations=='text':
        df_label = event_manager.extract_label_from_text_file(audio_dict['text'])
        data = event_manager.add_event_to_data(data,df_label)
    elif annotations == 'event':
        df_label = event_manager.extract_label_from_event_file(audio_dict['event'])
        data = event_manager.add_event_to_data(data,df_label)
    data_filtered = apply_filtering_on_signal(data, lf=120, hf= 1800, backward = True)
    return data_filtered,df_label