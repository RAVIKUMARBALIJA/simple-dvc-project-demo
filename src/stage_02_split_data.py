import pandas as pd
import argparse
import sys,os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.common_utils import read_params,create_dir,save_local_df,clean_prev_dir_if_exists
import logging
from sklearn.model_selection import train_test_split


logging_str = "[%(asctime)s: %(levelname)s: %(module)s)]: %(message)s"
logging.basicConfig(level=logging.DEBUG,format=logging_str)

def split_data(config_path):
    config = read_params(config_path)

    artifacts = config["artifacts"]

    raw_local_data = artifacts["raw_local_data"]
    split_data = artifacts["split_data"]

    processed_data_dir = split_data["processed_data_dir"]
    test_path = split_data["test_path"]
    train_path = split_data["train_path"]

    clean_prev_dir_if_exists(processed_data_dir)
    create_dir(dirs=[processed_data_dir,])

    base = config["base"]
    random_state = base["random_state"]
    target = base["target_col"]
    test_size = base["test_size"]

    df = pd.read_csv(raw_local_data,sep=",")

    train,test = train_test_split(df,test_size=test_size,random_state=random_state)

    for data,path in (train,train_path),(test,test_path):
        save_local_df(data,path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()

    try:
        split_data(parsed_args.config)
        logging.info(f"splitting data stage has been completed")
    except Exception as e:
        logging.error(e)
        traceback.print_exc()






    



