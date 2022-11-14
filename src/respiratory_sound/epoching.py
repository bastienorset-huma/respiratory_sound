import pandas as pd
import numpy as np


#Epoching

def get_epoching_from_label_data(data_feature,df_label,constant_values=0):
    epochs_list = []
    label_list = []
    for i in range(df_label.shape[0]):
        index = np.where((data_feature['time'] >= df_label.iloc[i].start) & (data_feature['time'] < df_label.iloc[i].end))[0]
        epoch_ = data_feature['data'][index]
        epochs_list.append(epoch_)
    try:
        epochs_final = np.stack(epochs_list) 
    except:
        epochs_final = convert_list_toarray(epochs_list,constant_values)
    return epochs_final

def get_epoching_from_label(data_feature,df_label,mean=True,constant_values=0):
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
        epochs_final = convert_list_toarray(epochs_list,constant_values)
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

def convert_list_toarray(list_data,constant_values=0):
    try:
        n_dim = list_data[0].ndim
    except:
        n_dim =1
    padding_len = get_max_length_from_array(list_data)
   
    for i,data_ in enumerate(list_data):
        pad_width = padding_len - data_.shape[n_dim-1]
        if n_dim==1:
            list_data[i] = np.pad(data_, pad_width=((0, pad_width)), mode='constant',constant_values=constant_values)
        elif n_dim==2:
            list_data[i] = np.pad(data_, pad_width=((0, 0), (0, pad_width)), mode='constant',constant_values=constant_values)
        elif n_dim==3:
            list_data[i] = np.pad(data_, pad_width=((0, 0), (0, 0), (0, pad_width)), mode='constant',constant_values=constant_values)
    if n_dim==2 or n_dim==1:
        array_data = np.stack(list_data)
    elif n_dim==3:
        array_data = np.concatenate(list_data)
    return array_data


def buffering_through_concatenation(df_label, epoch_data, win_len=0.100,win_shift=0.05,time_step=0.0125):
    df_label_new = pd.DataFrame(columns=['label','file'])
    X_list = list()
    label_list = []
    df_label['duration'] = df_label['end'] - df_label['start']
    buffer_length = int(win_len/time_step)
    buffer_shift = int(win_shift/time_step)
    for label in df_label.label.unique():
        index = df_label[df_label.label==label].index.tolist()
        X = np.concatenate(epoch_data[index,:,:],axis=1)
        X = X[:,~np.isnan(X).any(axis=0)]
        window = np.arange(0,X.shape[-1]-buffer_length,buffer_shift)
        for win in window:
            X_list.append(X[:,win:win+buffer_length])
            label_list.append(label)
    df_label_new['label'] = label_list
    df_label_new['file'] = df_label['file'].unique()[0]
    X_final = convert_list_toarray(X_list)
    return X_final, df_label_new