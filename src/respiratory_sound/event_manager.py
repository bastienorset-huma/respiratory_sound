
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
    df_label['file'] = text_file.split('/')[-1].split('.')[0]
    return df_label

def extract_label_from_event_file(audio_event_file,data):
    try:
        df_label = pd.read_csv(audio_event_file,delimiter='\t',header=None)
        df_label.columns = ['start','end','label']
        df_label['label'] = df_label['label'].str.replace(" ", "")
        df_label['label'] = df_label['label'].map({'wheeze':1,'crackle':2})
    except:
        df_label=pd.DataFrame.from_dict({
          'start': [0],
          'end': [data['time'][-1]],
        'label':[0]  
        })
    df_label['file'] = audio_event_file.split('/')[-1].split('_events')[0]
    return df_label

def add_event_to_data(data,df_label):
    data['label'] = np.zeros_like(data['time'],dtype=int)
    for i in range(df_label.shape[0]):
        index = np.where((data['time'] >=df_label['start'].iloc[i]) & (data['time'] <=df_label['end'].iloc[i]))[0]
        data['label'][index] = df_label['label'].iloc[i]
    return data

def fill_gap_event_frame(df_label,data_filtered):
    df_newlabel = pd.DataFrame()
    df_newlabel['start'] = pd.concat([df_label['start'],df_label['end']],axis=0)
    df_label['label'] = df_label['label'].astype(int) 
    df_newlabel = df_label.merge(df_newlabel,on='start',how='right')
    df_newlabel = df_newlabel.sort_values('start').reset_index(drop=True)
    df_newlabel['label'] = df_newlabel['label'].fillna(0)
    df_newlabel['end'] = df_newlabel['start'].shift(-1)
    if df_newlabel['start'].iloc[0] > 0:
        df_newlabel.loc[len(df_newlabel.index)] = [0, df_newlabel['start'].iloc[0], 0,df_newlabel['file'].iloc[0]]

    if df_newlabel['end'].iloc[0] < data_filtered['time'].max():
        df_newlabel['end'] = df_newlabel['end'].fillna(data_filtered['time'].max())
    df_newlabel = df_newlabel.sort_values('start').reset_index(drop=True)
    df_newlabel['duration'] = df_newlabel['end'] - df_newlabel['start']
    df_newlabel = df_newlabel[df_newlabel['duration'] > 0] 
    df_newlabel['file'] = df_newlabel['file'].iloc[0]
    return df_newlabel.drop(columns=['duration'])

def concat_annotation(df_label):
    if len(df_label['label'].unique())>1:
        print('more than 1 label')
        df_label['diff_label'] = df_label['label'].diff()
        df_label = df_label[df_label.diff_label !=0]
        df_label_new = pd.DataFrame(columns=['start','end','label'])
        df_label_new['start'] = df_label['start']
        df_label_new['end'] = df_label.start.shift(-1)
        df_label_new['end'].iloc[-1]= df_label['end'].iloc[-1]
        df_label_new['label'] = df_label['label']
    else: 
        df_label_new =  pd.DataFrame({
            'start': [df_label['start'].iloc[0]],
            'end':  [df_label['end'].iloc[-1]],
            'label': [df_label['label'].iloc[-1]]
        })
    return df_label_new

def segment_event_annotation(data,win_len=1,win_shift=0.5,threshold_class=[1,None,None]):
    def lambda_count_label(row):
        unique, counts = np.unique(row, return_counts=True)
        return dict(zip(unique, counts))

    times = data['time']
    times =  np.arange(0,times.max()-win_len,win_shift)

    df_segment = pd.DataFrame(columns=['start','end'])
    df_segment['start'] = times
    df_segment['end'] = df_segment['start'] + win_len

    data_dict = dict()
    for i in range(df_segment.shape[0]):
        index = np.where((data['time'] >= df_segment['start'].iloc[i]) & (data['time'] < df_segment['end'].iloc[i]))
        data_dict.update({df_segment['start'].iloc[i]:data['label'][index]})
        
    data_dict = pd.Series(data_dict).reset_index()
    data_dict.columns=['start','label']
    data_dict['label']  = data_dict['label'].apply(lambda x: lambda_count_label(x)) 
    data_dict = pd.concat([data_dict.drop(['label'], axis=1), data_dict['label'].apply(pd.Series)], axis=1)
    data_dict = data_dict.fillna(0)
    data_dict['total'] = data_dict[data_dict.columns[1:]].sum(axis=1)
    df_segment = df_segment.merge(data_dict,on='start',how='left')
    
    class_columns=df_segment.columns[2:-1].tolist()
    df_segment[class_columns] = df_segment[class_columns].div(df_segment['total'],axis=0)
    df_segment = df_segment.drop(columns='total')
    for class_ in class_columns:
        if threshold_class[class_]: 
            df_segment[class_] = np.where(df_segment[class_]>=threshold_class[class_],1,0)
        else:
            df_segment[class_] = np.where(df_segment[class_]!=0,1,0)
    df_segment['tot_label'] = df_segment[class_columns].sum(axis=1)
    df_segment = df_segment[df_segment.tot_label==1]
    df_segment['label'] = df_segment[class_columns].idxmax(axis=1)
    df_segment = df_segment.drop(columns=class_columns)
    df_segment = df_segment.drop(columns=['tot_label'])
    return df_segment.reset_index(drop=True)
    

def segment_event_annotation_noverlap(df_label,win_len=1,win_shift=0.5):
    df_label['duration'] = df_label['end'] - df_label['start']
    df_list_segment=list()
    for i in range(df_label.shape[0]):
        df_segment = pd.DataFrame(columns=['start','end','label'])
        if df_label['duration'].iloc[i] > win_len+win_len/2:
            times = np.arange(df_label.start.iloc[i],df_label.end.iloc[i]-win_len/2,win_shift)
            df_segment['start'] = times
            df_segment['end'] = df_segment['start'] + win_len
            df_segment['label'] = df_label['label'].iloc[i]
            df_segment['end'] = np.where(df_segment['end'] > df_label.end.iloc[i],df_label.end.iloc[i],df_segment['end']) #ensure until end of segment with padding
            
        else:
            df_segment = pd.DataFrame({
                'start':[df_label['start'].iloc[i]],
                'end':[df_label['end'].iloc[i]],
                'label':[df_label['label'].iloc[i]]
                })
        df_list_segment.append(df_segment)
    df_label_segmented = pd.concat(df_list_segment)
    df_label_segmented = df_label_segmented.reset_index(drop=True)
    df_label_segmented['duration'] = df_label_segmented['end'] - df_label_segmented['start']
    return df_label_segmented