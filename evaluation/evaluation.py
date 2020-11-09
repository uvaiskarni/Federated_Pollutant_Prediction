import os
import sys
import glob
sys.path.append('../preprocess/')
import utils
import numpy as np
import pandas as pd
from shutil import rmtree

ground_truth_dir = './test_data'


def smape(actual, predicted):
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)

    return 2 * np.mean(
        np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator != 0, casting='unsafe'))


def generate_groundtruth():
    if os.path.exists(ground_truth_dir):
        rmtree(ground_truth_dir)
    os.makedirs(ground_truth_dir)

    for id in utils.STATIONS:
        csv_files = glob.glob('../data/all_data/2019/*' + str(id) + "*.csv")
        data = pd.read_csv(csv_files[0])
        data = data[['Start'] + ["NO2", "NOX as NO2", "PM10", "PM2.5"]]
        data["Start"] = pd.to_datetime(data["Start"])
        data.set_index("Start", inplace=True)
        test_index = pd.to_datetime("2019-10-01 00:00:00")
        data = data[test_index:]
        data.to_csv(os.path.join(ground_truth_dir, str(id) + '.csv'))


def evaluate_avg(pred_dir='./Shenghui_centralized_model', ground_dir=ground_truth_dir):
    scores = []
    for id in utils.STATIONS:
        actual_data = pd.read_csv(os.path.join(ground_dir, str(id) + '.csv'))
        pred_data = pd.read_csv(os.path.join(pred_dir, str(id) + '.csv'))
        actual_data["Start"] = pd.to_datetime(actual_data["Start"])
        actual_data.set_index("Start", inplace=True)
        pred_data = pred_data[['Start'] + ["NO2", "NOX as NO2", "PM10", "PM2.5"]]
        pred_data["Start"] = pd.to_datetime(pred_data["Start"])
        pred_data.set_index("Start", inplace=True)
        pred_data[pred_data < 0] = 0
        values = [smape(act[1], pred[1]) for act, pred in zip(actual_data.iterrows(), pred_data.iterrows()) if
                  not any(act[1] < 0) and not act[1].isnull().values.any()
                  ]
        scores.append((np.mean(values), len(values)))
    print(scores)
    print(sum(i * j for i, j in scores) / sum(j for _, j in scores))


if __name__ == "__main__":
    # generate_groundtruth()
    evaluate_avg()
