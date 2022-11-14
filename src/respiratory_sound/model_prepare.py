
import pickle 
import sklearn 
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

import epoching

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
    
def select_nsample_normal(df_label,max_sample=50):
    if len(df_label.label.unique()) > 1:
        max_sample = df_label.label[df_label.label!=0].value_counts().max()
        df_label_0 = df_label[df_label.label==0].iloc[:max_sample]
        max_sample=df_label.label[df_label.label!=0].value_counts().max()
        df_label_o = df_label[df_label.label!=0]
        df_label_list = pd.concat([df_label_o,df_label_0],axis=0)
    else:
        df_label_list = df_label.iloc[:max_sample]
    df_label_list = df_label_list.reset_index(drop=True)
    return df_label_list

def prepare_data_time_for_modeling(epochs_list,df_label_list):
    epoch_final = epoching.convert_list_toarray(epochs_list)
    df_all_label = pd.concat(df_label_list)
    df_all_label = df_all_label.reset_index(drop=True)
    return epoch_final, df_all_label
    
def prepare_data_for_modeling(epochs_list,df_label_list):
    df_all_label = pd.concat(df_label_list)
    df_all_label = df_all_label.reset_index(drop=True)
    X = np.concatenate(epochs_list)
    X = pd.concat([pd.DataFrame(X),df_all_label],axis=1)
    X.columns = X.columns.astype(str)
    return X

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
        
def prepare_feature_for_epoching(data_filtered):
    feature,_ = extract_features(data_filtered)
    data_feature = {
            'data': np.concatenate([col for col in feature.values()]),
            'time': librosa.times_like(feature['mfcc'], sr=data_filtered['fs']),
        }
    return data_feature

def convert_list_toarray(list_data,padding_len):
    for i,data_ in enumerate(list_data):
        pad_width = padding_len - data_.shape[1]
        list_data[i] = np.pad(data_, pad_width=((0, 0), (0, pad_width)), mode='constant')
    array_data = np.stack(list_data)
    return array_data