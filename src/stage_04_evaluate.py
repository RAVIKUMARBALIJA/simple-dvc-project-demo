import pandas as pd
import argparse
import sys,os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common_utils import read_params,create_dir,save_reports,load_model,generate_metrics
import logging


logging_str = "[%(asctime)s,%(levelname)s,%(module)s)]: %(message)s"
logging.basicConfig(level=logging.DEBUG,format=logging_str)

def evaluate_model(config_path):

    config = read_params(config_path)

    artifacts = config["artifacts"]

    model_path = artifacts["model_path"]

    split_data = artifacts["split_data"]
    test_path = split_data["test_path"]

    base = config["base"]
    target = base["target_col"]

    report = artifacts["report"]
    scores_file = report["scores"]

    test = pd.read_csv(test_path,sep=",")
    test_y = test[target]
    test_x = test.drop(target,axis=1)

    model = load_model(model_path)

    predict_y = model.predict(test_x)

    rmse,mae,r2 = generate_metrics(test_y,predict_y)

    scores = {"rmse": rmse,
              "mae": mae,
              "r2": r2}
    save_reports(scores_file,scores)

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()

    try:
        evaluate_model(parsed_args.config)
        logging.info(f"model evaluation is complete")
    except Exception as e:
        logging.error(e)
        traceback.print_exc()