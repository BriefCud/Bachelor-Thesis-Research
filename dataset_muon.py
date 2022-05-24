import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
DATA_PATH = 'data/'
SCALING_FACTOR = np.pi

def load_dataset(train_size,test_size,seed=1):
    assert (train_size % 2) == 0 and (test_size % 2) == 0 # Must be even to have a balanced dataset

    #features_cols = ['mu_Q', 'mu_pTrel', 'mu_dist', 'Jet_QTOT'] # "Muon Dataset" features
    target_col = 'Jet_LABEL' # Monte Carlo truth 0 -> b     1 -> b-bar  

    # Jets are split in two CSVs for training and testing 
    train_csv = pd.read_csv(DATA_PATH + 'trainData.csv')
    test_csv = pd.read_csv(DATA_PATH + 'testData.csv')

    # Selecting only the jets with a muon
    train_csv = train_csv[train_csv['mu_Q'] != 0]
    test_csv = test_csv[test_csv['mu_Q'] != 0]

    # Features are normalized to the [0,1] range
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([train_csv,test_csv])[features_cols])

    # Sample randomly a balanced set of jets for training and testing
    train_sample = pd.concat([train_csv[train_csv[target_col] == 0].sample(n=train_size//2),train_csv[train_csv[target_col] == 1].sample(n=train_size//2,random_state=seed)])
    test_sample = pd.concat([test_csv[test_csv[target_col] == 0].sample(n=test_size//2),test_csv[test_csv[target_col] == 1].sample(n=test_size//2,random_state=seed)])


    train_features = scaler.transform(train_sample[features_cols])*SCALING_FACTOR  # Features are mapped to angles so I scale them to [0,pi]
    train_target = train_sample[target_col]*2-1 # Monte Carlo truth is scaled to  -1 -> b      +1 -> b-bar

    # Same for testing dataset
    test_features = scaler.transform(test_sample[features_cols])*SCALING_FACTOR
    test_target = test_sample[target_col]*2-1

    return train_features,train_target,test_features,test_target
