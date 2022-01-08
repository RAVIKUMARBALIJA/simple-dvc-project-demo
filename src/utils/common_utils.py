import os
import json
import yaml
import logging
import shutil
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def read_params(config_path: str) -> dict:
    with open(config_path,"r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    logging.info(f"reading the param")
    return config

def create_dir(dirs:list):
    for dir in dirs:
        os.makedirs(dir,exist_ok=True)
        logging.info(f"created directory at {dir}")

def save_local_df(df,df_path,header=False):
    if header:
        new_cols = [col.replace(" ","_") for col in df.columns]
        df.to_csv(df_path,index=False,header=new_cols)
        logging.info(f"dataframe saved at {df_path}")
    else:
        df.to_csv(df_path,index=False)
        logging.info(f"dataframe saved at {df_path}")

def save_reports(report_path:str,report:dict):
    with open(report_path,"w") as f:
        json.dump(report_path,f,indent=4)
    logging.info(f"details of the report {report}")
    logging.info(f"saved report at {report_path}")

def clean_prev_dir_if_exists(dirpath:str):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
        logging.info(f"clearing existing artifacts at {dirpath}")

def save_model(model_obj,model_path):
    with open(model_path,"w") as f:
        pickle.dump(model_obj,f)

def load_model(model_path):
    return pickle.load(open(model_path,"r"))

def generate_metrics(actual,predicted):
    rmse = np.sqrt(mean_squared_error(actual,predicted))
    mae = mean_absolute_error(actual,predicted)
    r2 = r2_score(actual,predicted)
    return rmse,mae,r2
