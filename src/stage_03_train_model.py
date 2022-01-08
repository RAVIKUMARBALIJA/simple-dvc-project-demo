import pandas as pd
import argparse
from src.utils.common_utils import read_params,create_dir,save_reports,save_model
import logging
from sklearn.linear_model import ElasticNet

def train(config_path):
    config = read_params(config_path)

    artifacts = config["artifacts"]

    split_data = artifacts["split_data"]
    train_path = split_data["train_path"]

    model_dir = artifacts["model_dir"]
    model_path = artifacts["model_path"]

    report = artifacts["report"]
    report_dir = report["report_dir"]
    params_file = report["params"]
    scores_file = report["scores"]

    base = config["base"]
    target = base["target_col"]

    create_dir(dirs = [model_dir,report_dir])

    train_df = pd.read_csv(train_path,sep = ",")
    train_x = train_df.drop(target,axis=1)
    train_y = train_df[target]

    elasticnet_params = config["estimators"]["ElasticNet"]["params"]
    alpha = elasticnet_params["alpha"]
    l1_ratio = elasticnet_params["l1_ratio"]

    elasticNet = ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
    elasticNet.fit(train_x,train_y)

    params = {"alpha":alpha,"l1_ratio":l1_ratio}
    save_reports(params_file,params)
    save_model(elasticNet,model_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()

    try:
        train(parsed_args.config)
        logging.info(f"model training has been completed")
    except Exception as e:
        logging.error(e)




