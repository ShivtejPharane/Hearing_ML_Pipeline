import os
import pandas as pd
import logging
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import yaml
from dvclive import Live
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log_dir = os.path.join(PROJECT_ROOT,"logs")
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

def train_model(x_train:np.ndarray,y_train:np.ndarray,max_depth:int)->DecisionTreeClassifier:
    """Train the Descision tree model 
    x_train : Training features
    y_train : Training label
    """
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in x_train and y_train must be same.")
        logger.debug("Intialize the Decision tree With Parameters ")
        dt = DecisionTreeClassifier(max_depth=max_depth)

        logger.debug("Model Training Started with %d smaples",x_train.shape[0])
        dt.fit(x_train,y_train)
        logger.debug("Model Training completed")

        return dt
    except ValueError as e:
        logger.error("Value Error during the model training :%s",e)
        raise
    except Exception as e:
        logger.error("Unexpected Error : %s",e)
        raise
def save_model(model,file_path:str)->None:
    """Save the trained model to a file"""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug("model saved to %s",file_path)
    except FileNotFoundError as e:
        logger.debug('File path not found %s',e)
    except Exception as e:
        logger.debug('Unexpected error %s',e)


def main():
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        params = load_params('params.yaml')['model_training']['max_depth']
        x_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        
        dt = train_model(x_train=x_train,y_train=y_train,max_depth=params)
        
        model_save_path = os.path.join('.','models','model.pkl')
        save_model(dt,model_save_path)
        
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
        



