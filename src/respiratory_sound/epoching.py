import pandas as pd
import numpy as np


#Epoching
def get_epoching_from_label(data_feature,df_label,mean=True):
    epochs_list = []
    label_list = []
    for i in range(df_label.shape[0]):
        label_ = df_label.iloc[i].label.astype(int)
        index = np.where((data_feature['time'] >= df_label.iloc[i].start) & (data_feature['time'] < df_label.iloc[i].end))[0]
        if len(index) > 0:
            epoch_ = data_feature['data'][:,index]
            if mean:
                epochs_list.append(np.mean(epoch_,axis=1))
            else:
                epochs_list.append(epoch_)
            label_list.append(label_)
    if mean:
        epochs_final = np.stack(epochs_list)    
    else:
        epochs_final = convert_list_toarray(epochs_list)
    return epochs_final,label_list

def stacking_data_based_on_dict(list_data,df_label):
    epochs_label = dict.fromkeys(df_label.label.unique().tolist(), [])
    for key in epochs_label.keys():
        index = np.where(df_label['label'] == key)[0]
        data = [e for i,e in enumerate(list_data) if i in index]
        epochs_label[key] = np.stack(data)
        return epochs_label

def get_max_length_from_array(list_data):
    padding_len = 0
    n_dim = list_data[0].ndim
    for data_ in list_data:
        if padding_len < np.shape(data_)[n_dim-1]:
                padding_len = np.shape(data_)[n_dim-1]
    return padding_len

def convert_list_toarray(list_data):
    n_dim = list_data[0].ndim
    padding_len = get_max_length_from_array(list_data)
    for i,data_ in enumerate(list_data):
        pad_width = padding_len - data_.shape[n_dim-1]
        if n_dim==2:
            list_data[i] = np.pad(data_, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif n_dim==3:
            list_data[i] = np.pad(data_, pad_width=((0, 0), (0, 0), (0, pad_width)), mode='constant')
    if n_dim==2:
        array_data = np.stack(list_data)
    elif n_dim==3:
        array_data = np.concatenate(list_data)
    return array_data