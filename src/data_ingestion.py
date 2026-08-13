import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log_dir = os.path.join(PROJECT_ROOT,"logs")
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_ingetion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url : str)-> pd.DataFrame : 
    """Load the csv file"""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data Loaded from %s",data_url)
        return df
    except pd.errors as e :
        logger.debug("Failed to parse the csv :%s",e)
        raise
    except Exception as e:
        logger.debug("Unexcepted error occured during the load the data")
        raise

def save_data(file_path:str,df:pd.DataFrame):
    try:
        dataset_path=os.path.join(file_path,'Dataset')
        os.makedirs(dataset_path,exist_ok=True)
        df.to_csv(os.path.join(dataset_path,'dataset.csv'),index=False)
        logger.debug("dataset saved to %s",file_path)
    except Exception as e:
        logger.debug("failed to the save Dataset")
        raise
    

def main():
    try:
        test_size=0.2
        data_path = "https://raw.githubusercontent.com/ShivtejPharane/Datasets/refs/heads/main/hearing_test.csv"
        df = load_data(data_path)
        save_data("dataset",df)
    except Exception as e:
        logger.debug("Failed to the data ingestion process")
        print(f"error {e}")

if __name__=='__main__':
    main()

