import os 
import numpy as np
import librosa
import librosa.display
from pyparsing import col
from scipy import signal

import utils

import event_manager

SAMPLING_RATE_DEFAULT = 4000

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

def apply_melspec(data,n_fft=2048,n_mel=None,win_shift=None,center=False,normalize=True):
    if not win_shift:
        win_shift = int(n_fft/4)
    if n_mel:
        X = librosa.feature.melspectrogram(y=data['data'], sr=data['fs'],n_fft=n_fft,hop_length=win_shift, n_mels=n_mel,center=center)
    else:
        X = librosa.feature.melspectrogram(y=data['data'], sr=data['fs'],n_fft=n_fft,hop_length=win_shift,center=center)
    X = librosa.power_to_db(X, ref=np.max)

    if normalize:
        mel_min = np.min(X)
        mel_max = np.max(X)
        diff = mel_max - mel_min
        if diff > 0:
            X = (X - mel_min) / diff
        else:
            print('issue divide by 0')
    
    times = librosa.times_like(X,sr=data['fs'],n_fft=n_fft,hop_length=win_shift)
    return {
        'data': X,
        'time': times,
        'fs':data['fs']
    }

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
    data_wv['data'] = 20*np.log10(np.abs(cs1))
    data_wv['freq'] = f1[::-1]
    return data_wv,f1

def apply_wavelets_on_epochs(epochs_data,fs=4000,normalize=True):
    wv_data = []
    for i in range(epochs_data.shape[0]):
        y = epochs_data[i,:]
        cs1, f1 = utils.cwt2(y, nv=12, sr=fs)
        xx = 20*np.log10(np.abs(cs1))
        if normalize:
            xx = xx-xx.min()/(xx.max() - xx.min())
        wv_data.append(xx)
    return np.stack(wv_data),f1

def preprocess_data(audio_dict, fs=SAMPLING_RATE_DEFAULT,lf=120,hf=1800, annotations='text'):
    data = extract_data_from_file(audio_dict['data'],fs)
    if annotations=='text':
        df_label = event_manager.extract_label_from_text_file(audio_dict['text'])
        data = event_manager.add_event_to_data(data,df_label)
    elif annotations == 'event':
        df_label = event_manager.extract_label_from_event_file(audio_dict['event'],data)
        data = event_manager.add_event_to_data(data,df_label)
    data_filtered = apply_filtering_on_signal(data, lf=lf, hf= hf, backward = True)
    return data_filtered,df_label