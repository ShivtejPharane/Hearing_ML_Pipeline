import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import logging
from pathlib import Path


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = Path(__file__).resolve().parent

log_dir = os.path.join(PROJECT_ROOT,"logs")
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = "./logs/model_evaluation.py"
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_model(file_path:str):
    """Load The trained Model"""
    try : 
        with open(file_path,'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s',file_path)
        return model
    except FileNotFoundError:
        logger.error('File Not found %s',file_path)
        raise
    except Exception as e:
        logger.error('Unexcepted error occured while loading the model : %s',e)
        raise

def load_data(file_path:str)-> pd.DataFrame:
    """Load data from the csv file"""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s",file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file : %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error is occured while loading the data : %s',e)
        raise

def evaluate_model(dt,x_test:np.ndarray,y_test:np.ndarray)->dict:
    """Evaluate the model and return the evaluation metrics"""
    try:
        y_pred = dt.predict(x_test)
        y_pred_proba = dt.predict_proba(x_test)[:,1]

        accuracy = accuracy_score(y_test,y_pred=y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred)

        matrics_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'auc' : auc
        }
        logger.debug('Model evaluation metrics calculated')
        return matrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s',e)
        raise

def save_metrics(metrics:dict,file_path:str)->None :
    """Save the evaluation metrics to the json file"""
    try:
        # Ensure the directory exist
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
            logger.debug("Metrices Saved to %s",file_path)
    except Exception as e:
        logger.error('Error Occured while saving the metrics : %s',e)
        raise
def main():
    try:
        model_path = Path(PROJECT_ROOT) / "models" / "model.pkl"
        test_data_path = Path(PROJECT_ROOT) / "data" / "raw" / "test.csv"
        reports_path = Path(PROJECT_ROOT) / "reports" / "metrics.json"
        #params = load_params(params_path='params.yaml')
        clf = load_model(model_path)
        test_data = load_data(test_data_path)

        x_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values

        metrics = evaluate_model(clf,x_test,y_test)

        # Expriment tracking using dvc live 
        # with Live(save_dvc_exp=True) as live:
        #     live.log_metric('accuracy',accuracy_score(y_test,y_test))
        #     live.log_metric('pricision',precision_score(y_test,y_test))
        #     live.log_metric('recall',recall_score(y_test,y_test))

            #live.log_params(params)
        save_metrics(metrics,reports_path)
    except Exception as e:
        logger.error('Failed to complete the model evaluation process : %s',e)
        print(f"Error : {e}")

if __name__ == '__main__':
    main()

