import pandas as pd
import pickle
import numpy as np
import sklearn

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

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


def get_class_weight(y,return_dict=False):
    classes_weight = class_weight.compute_sample_weight(class_weight='balanced',y=y)
    if return_dict:
        classes_weight = dict(zip(np.unique(y), classes_weight))
    return classes_weight

def structure_data_for_modeling(file,classes=[0,1,2]):    
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
        
    X,df_label = data['feature'],data['label']
    df_label = df_label[df_label.label.isin(classes)]
    X=X[df_label.index.tolist(),:,:]
    if classes==[0,2]:
        df_label['label'] = df_label['label'].replace(2,1)
    return X, df_label
    
def do_train_test_split_based_onsubject(X,df_label,reshape_data=True,n_subject=100):
    df_label['Subject_ID'] = df_label['file'].str.split('_').str[0]
    SubjectID = sklearn.utils.shuffle(df_label.Subject_ID.unique(),random_state=42)
    SubjectID_train = SubjectID[:n_subject]
    df_label['train_test_split'] = np.where(df_label['Subject_ID'].isin(SubjectID_train),'train','test')
    index_train= np.where(df_label.train_test_split == 'train')[0]
    index_test = np.where(df_label.train_test_split == 'test')[0]
    X_train = X[index_train,:]
    X_test = X[index_test,:]
    y_train = df_label.loc[df_label['train_test_split'] == 'train','label'].to_numpy()
    y_test = df_label.loc[df_label['train_test_split'] == 'test','label'].to_numpy()
    
    if reshape_data:
        X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))

    return X_train,X_test,y_train,y_test,df_label