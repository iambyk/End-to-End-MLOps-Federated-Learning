import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

def load_and_partition_data(file_path, num_clients=3):
    df = pd.read_csv(file_path)
    
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    sss = StratifiedShuffleSplit(n_splits=num_clients, test_size=1/num_clients, random_state=42)
    
    client_data = []
    for train_index, test_index in sss.split(X, y):
        X_client, y_client = X[test_index], y[test_index]
        
    
        inner_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in inner_sss.split(X_client, y_client):
            X_train, X_test = X_client[train_idx], X_client[test_idx]
            y_train, y_test = y_client[train_idx], y_client[test_idx]
            client_data.append((X_train, y_train, X_test, y_test))
            break
            
    return client_data