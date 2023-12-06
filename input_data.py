# Libraries

import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import configparser
config = configparser.ConfigParser()
config.read("config.ini")

train_file_path = config['INPUT_DATA']['TRAIN_PATH']
test_file_path = config['INPUT_DATA']['TEST_PATH']

def input_data():
    # Read data
    train_df  = pd.read_csv(train_file_path, delimiter = "\t", header=None)
    test_df  = pd.read_csv(test_file_path, delimiter = "\t", header=None)
    train_df.columns = ["category","patient_summary"]
    test_df.columns = ["patient_summary"]
    print(train_df.shape, test_df.shape)
    return train_df, test_df


