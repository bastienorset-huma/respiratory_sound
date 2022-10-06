import os 

import pandas as pd
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment
from scipy import signal

import utils


SAMPLING_RATE_DEFAULT = 4000

def load_file_from_recording_name(audio_folder,audio_subject):
    audio_data_file = f'{audio_subject}.wav'
    audio_event_file = f'{audio_subject}_events.txt'
    audio_data_file = os.path.join(audio_folder,audio_data_file)
    audio_event_file = os.path.join(audio_folder,audio_event_file)
    return audio_data_file, audio_event_file
    
def get_info_from_file(file):
    audio_info = dict()
    audio_segment = AudioSegment.from_file(file)
    audio_info.update({'n_channel': audio_segment.channels})
    audio_info.update({'sample_width': audio_segment.sample_width})
    audio_info.update({'fs': audio_segment.frame_rate})
    audio_info.update({'frame_width': audio_segment.frame_width})
    audio_info.update({'length (ms)': len(audio_segment)})
    audio_info.update({'n_frames': audio_segment.frame_count()})
    audio_info.update({'intensity': audio_segment.dBFS})
    return audio_info

def extract_data_from_file(file,fs= SAMPLING_RATE_DEFAULT):
    x_data, sr = librosa.load(file, sr=fs)
    audio_data = {
        'data': x_data,
        'time': np.linspace(0,len(x_data)/sr,len(x_data))
        }
    audio_data['fs'] = sr
    return audio_data

def add_event_to_data(data,event_file):
    try:
        df_label = pd.read_csv(event_file,delimiter='\t',header=None)
        df_label.columns = ['start','end','label']
        df_label['label'] = df_label['label'].map({'wheeze':1,'crackle':2})
        data['label'] = np.zeros_like(data['time'])
        for i in range(df_label.shape[0]):
            index = np.where((data['time'] >=df_label['start'].iloc[i]) & (data['time'] <=df_label['end'].iloc[i]))[0]
            data['label'][index] = df_label['label'].iloc[i]
    except:
        data['label'] = np.zeros_like(data['time'])
    return data
    
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


def preprocess_data(audio_data_file, audio_event_file):
    data = extract_data_from_file(audio_data_file)
    data = add_event_to_data(data,audio_event_file)
    data_filtered = apply_filtering_on_signal(data, lf=120, hf= 1800, backward = True)
    return data_filtered

def epoching_based_on_label(data):
    label_data_dict = dict()
    label_list = np.unique(data['label'])
    for label_ in label_list:
        index_time = np.where(data['label'] == label_)[0]
        if data['data'].ndim == 1:
            label_data = data['data'][index_time]
        elif data['data'].ndim == 2:
            label_data = data['data'][:,index_time]
        label_data_dict.update({label_: label_data})
        
    return label_data_dict

def apply_wavelets(data):
    cs1, f1 = utils.cwt2(data['data'], nv=12, sr=data['fs'],low_freq=60)
    data_wv = data.copy()
    data_wv['data'] = np.log10(np.abs(cs1))
    data_wv['freq'] = f1[::-1]
    return data_wv

def extract_features(data):
    feature = {
        'rmse': librosa.feature.rms(y=data['data']),
        'chroma_stft': librosa.feature.chroma_stft(y=data['data'], sr=data['fs']),
        'spec_cent': librosa.feature.spectral_centroid(y=data['data'], sr=data['fs']),
        'spec_bw': librosa.feature.spectral_bandwidth(y=data['data'], sr=data['fs']),
        'rolloff': librosa.feature.spectral_rolloff(y=data['data'], sr=data['fs']),
        'zcr': librosa.feature.zero_crossing_rate(data['data']),
        'mfcc': librosa.feature.mfcc(y=data['data'], sr=data['fs']),
        }
    feature['delta_mfcc'] = librosa.feature.delta(feature['mfcc'])
    feature['delta2_mfcc'] = librosa.feature.delta(feature['mfcc'],order=2)
    
    list_feature_type = []
    for k, v in feature.items():
        list_feature_type.append([k]*v.shape[0])
    list_feature_type = np.concatenate(list_feature_type)
    return feature,list_feature_type
    
def prepare_features_ml(list_feature,mean_data=True):
    df = pd.DataFrame(list_feature)
    MAX_LEN = df['mfcc'].apply(lambda x : x.shape[1]).max()
    df_concatenate = df.apply(lambda x: np.concatenate(x), axis=1)
    if mean_data:
        df_mean_overtime = df_concatenate.apply(lambda x: x.mean(axis=1))
        features = np.array(df_mean_overtime.values.tolist())
    else:
        features = convert_list_toarray(df_concatenate,padding_len=MAX_LEN)
    return features
        
def convert_list_toarray(list_data,padding_len):
    for i,data_ in enumerate(list_data):
        pad_width = padding_len - data_.shape[1]
        list_data[i] = np.pad(data_, pad_width=((0, 0), (0, pad_width)), mode='constant')
    array_data = np.stack(list_data)
    return array_data