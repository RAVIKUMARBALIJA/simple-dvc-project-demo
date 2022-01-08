import pandas as pd
import argparse
from src.utils.common_utils import read_params,create_dir,save_local_df,clean_prev_dir_if_exists

import logging


def get_data(config_path):
    config = read_params(config_path)

    data_path = config["data_source"]["s3_source"]
    artifacts_dir = config["artifacts"]

    raw_local_data_dir = artifacts_dir["raw_local_data_dir"]
    raw_local_data = artifacts_dir["raw_local_data"]

    clean_prev_dir_if_exists(artifacts_dir)
    create_dir(dirs=[artifacts_dir,raw_local_data_dir])

    df = pd.read_csv(data_path)

    save_local_df(df,raw_local_data,header=True)

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()

    try:
        get_data(parsed_args.config)
        logging.info(f"reading and writing stage has been completed")
    except Exception as e:
        logging.error(e)




