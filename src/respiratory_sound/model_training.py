import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

import epoching

def prepare_data_time_for_modeling(epochs_list,label_list,list_audio_files):
    epoch_final = epoching.convert_list_toarray(epochs_list)
    list_label = []
    list_file = []
    for label_,file in zip(label_list,list_audio_files):
        list_file.append([file]*len(label_))
    df_label = pd.DataFrame(
        {'label': np.concatenate(label_list), 
        'file': np.concatenate(list_file)
        })
    return epoch_final, df_label
    
def prepare_data_for_modeling(epochs_list,label_list,list_audio_files):
    feature = []
    list_label = []
    list_file = []
    for epochs_,label_,file in zip(epochs_list,label_list,list_audio_files):
        feature.append(epochs_)
        list_label.append(label_)
        list_file.append([file]*epochs_.shape[0])
        
    X = np.concatenate(feature)
    y = np.concatenate(list_label)
    file_data = np.concatenate(list_file)
    X = pd.DataFrame(X)
    X['label'] = y
    X['file'] = file_data
    X.columns = X.columns.astype(str)
    return X

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

def apply_smote_for_balancing_dataset(data): 
    X = data.drop(columns='label').to_numpy()
    y = data['label'].to_numpy()
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    new_data = pd.DataFrame(X_res)
    new_data['label'] = y_res
    new_data.columns = new_data.columns.astype(str)
    return new_data
    
def select_n_sample(df,n):
    df_concat = list()
    for label in df.label.unique():
        df_label = df[df.label==label]
        if n > df_label.shape[0]:
            df_concat.append(df_label)
        else:
            df_concat.append(df_label.iloc[:n])
    return pd.concat(df_concat)
        
def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) # return accuracy score