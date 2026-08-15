import os
import logging
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import yaml
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log_dir = os.path.join(PROJECT_ROOT,"logs")
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def dataset_prepration(df:pd.DataFrame):
    "Split the Data in x and y"
    try:
        print(df.head())
        x=df.drop(['test_result'],axis=1)
        y= df['test_result']
        logger.debug("Data prepration done")
        return x,y
    except Exception as e:
        logger.debug("Unexpected Error : %s",e)
        raise


def overSampling(x:pd.DataFrame,y:pd.DataFrame):
    """Done the Over Sampling for the Imbalanced dataset """
    try:
        smote = SMOTE()
        x,y=smote.fit_resample(x,y)
        logger.debug("over Sampling Done")
        return x,y
    except Exception as e:
        logger.debug("Unexpected error %s",e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    """Save the data"""
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug("Train and Test data saved to %s",raw_data_path)
    except Exception as e:
        logger.debug("Unecpected error occured : %s",e)
        raise  


def main():
    params = load_params(params_path='params.yaml')
    test_size = params['data_preprocessing']['test_size']
    df = pd.read_csv("./dataset/Dataset/dataset.csv")
    x,y = dataset_prepration(df)
    sampled_x,sampled_y = overSampling(x,y)
    sampled_df = pd.concat([sampled_x,sampled_y],axis=1)
    train_data,test_data = train_test_split(sampled_df,test_size=test_size,random_state=42)
    save_data(train_data,test_data,data_path='./data')

if __name__ == '__main__':
    main()