
import pandas as pd
import numpy as np

def extract_label_from_text_file(text_file):
    df_label = pd.read_csv(text_file,delimiter='\t',header=None) 
    df_label.columns=['start','end','crackle','wheeze']
    df_label['label'] = np.where(df_label['crackle']==1,'crackle','normal')
    df_label['label'] = np.where(df_label['wheeze']==1,'wheeze',df_label['label'])
    df_label['label'] = np.where(df_label[['crackle','wheeze']].sum(axis=1)==2,'both',df_label['label'])
    df_label.drop(columns=['crackle','wheeze'],inplace=True)
    df_label['label'] = df_label['label'].map({'wheeze':1,'crackle':2,'both':3,'normal':0})
    return df_label

def extract_label_from_event_file(audio_event_file):
    try:
        df_label = pd.read_csv(audio_event_file,delimiter='\t',header=None)
        df_label.columns = ['start','end','label']
        df_label['label'] = df_label['label'].map({'wheeze':1,'crackle':2})
    except:
        df_label=pd.DataFrame(columns=['start','end','label'])
    return df_label

def add_event_to_data(data,df_label):
    data['label'] = np.zeros_like(data['time'],dtype=int)
    for i in range(df_label.shape[0]):
        index = np.where((data['time'] >=df_label['start'].iloc[i]) & (data['time'] <=df_label['end'].iloc[i]))[0]
        data['label'][index] = df_label['label'].iloc[i]
    return data

def fill_gap_event_frame(audio_event_file,data_filtered):
    try:
        df_newlabel = pd.DataFrame()
        df_label = extract_label_from_event_file(audio_event_file)
        df_newlabel['start'] = pd.concat([df_label['start'],df_label['end']],axis=0)
        df_label['label'] = df_label['label'].astype(int) 
        df_newlabel = df_label.merge(df_newlabel,on='start',how='right')
        df_newlabel = df_newlabel.sort_values('start').reset_index(drop=True)
        df_newlabel['label'] = df_newlabel['label'].fillna(0)
        df_newlabel['end'] = df_newlabel['start'].shift(-1)
        df_newlabel.loc[len(df_newlabel.index)] = [0, df_newlabel['start'].iloc[0], 0]
        df_newlabel['end'] = df_newlabel['end'].fillna(data_filtered['time'].max())
        df_newlabel = df_newlabel.sort_values('start').reset_index(drop=True)
    except:
        df_newlabel = pd.DataFrame(columns=['start','end','label'])
        df_newlabel.loc[len(df_newlabel.index)] = [0, data_filtered['time'].max(), 0]
    return df_newlabel

def segment_event_annotation(data,times):
    win_len = times[0] * 2
    times =  times -   times[0]
    df_segment = pd.DataFrame(columns=['start','end'])
    df_segment['start'] = times
    df_segment['end'] = df_segment['start'] + win_len
    data_dict = dict()
    for i in range(df_segment.shape[0]):
        index = np.where((data['time'] >= df_segment['start'].iloc[i]) & (data['time'] < df_segment['end'].iloc[i]))
        data_dict.update({df_segment['start'].iloc[i]:data['label'][index]})
        
    data_dict = pd.Series(data_dict).reset_index()
    data_dict.columns=['start','label']
    data_dict['label'] = data_dict['label'].apply(lambda x: np.bincount(x).argmax())
    df_segment = df_segment.merge(data_dict,on='start',how='left')
    return df_segment