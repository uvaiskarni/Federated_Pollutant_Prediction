import os
import sys

sys.path.append('../preprocess/')
sys.path.append('../evaluation/')
import utils
from utils import *
import tensorflow as tf
import numpy as np
from dataset_generator import WindowGenerator
import matplotlib.pyplot as plt
import pandas as pd
from shutil import rmtree

# %%

stations = {}
for id in utils.STATIONS:
    data = pd.read_csv(os.path.join(utils.PRE_DIR, str(id) + '.csv'))
    data["Start"] = pd.to_datetime(data["Start"])
    data = data[["Start"] + INPUTS]
    data.set_index("Start", inplace=True)
    print(data.isnull().sum())
    stations[id] = data

# %%

test_index = pd.to_datetime("2019-10-01 00:00:00")
val_index = pd.to_datetime("2019-6-01 00:00:00")
s = list(stations.values())[0]

# %%

datasets = []
for id in stations:
    station = stations[id]
    n = len(station)
    # print(id)
    train_df = station[:val_index]
    val_df = station[val_index: test_index]
    test_df = station[test_index:]
    datasets.append({'train_set': train_df, "val_set": val_df, "test_set": test_df})

# %%

train_all = pd.concat([dataset["train_set"].reset_index() for dataset in datasets])
train_mean = train_all.mean()
train_std = train_all.std()

datasets = [{'train_set': ((station['train_set'] - train_mean) / train_std),
             'val_set': (station['val_set'] - train_mean) / train_std,
             'test_set': (station['test_set'] - train_mean) / train_std
             } for station in datasets]

# %%

multi_window = WindowGenerator(input_width=IN_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               train_df=[d['train_set'] for d in datasets],
                               test_df=[d['test_set'] for d in datasets],
                               val_df=[d['val_set'] for d in datasets],
                               label_columns=utils.OUTPUTS)

multi_window.plot(cols=utils.OUTPUTS)

# %%

multi_val_performance = {}
multi_performance = {}

MAX_EPOCHS = 30


def compile_and_fit(model, window, patience=10, recompile=True):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    def smape(y_true, y_pred):
        y_true = y_true * train_std[:NUM_OUTPUTS] + train_mean[:NUM_OUTPUTS]
        y_pred = y_pred * train_std[:NUM_OUTPUTS] + train_mean[:NUM_OUTPUTS]
        return tf.reduce_mean(2 * tf.abs(y_true - y_pred)
                              / (tf.abs(y_pred) + tf.abs(y_true)), axis=-1)

    if recompile:
        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[smape])
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val, verbose=1,
                        callbacks=[early_stopping])
    return history


# %%

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(64, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS * NUM_OUTPUTS,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, NUM_OUTPUTS])
])

history = compile_and_fit(multi_lstm_model, multi_window)

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=1)
multi_window.plot(multi_lstm_model, cols=utils.OUTPUTS)

# %%

x = np.arange(len(multi_performance))
width = 0.3

val_smape = [v[1] for v in multi_val_performance.values()]
test_smape = [v[1] for v in multi_performance.values()]

plt.bar(x - 0.17, val_smape, width, label='Validation')
plt.bar(x + 0.17, test_smape, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'SMAP (average over all times and outputs)')
_ = plt.legend()


# %%

def smape(actual, predicted):
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)

    return 2 * np.mean(
        np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator != 0, casting='unsafe'))


pred_dir = "../evaluation/Shenghui_centralized_model/"
if os.path.exists(pred_dir):
    rmtree(pred_dir)
os.makedirs(pred_dir)

smapes = []
for id in stations:
    input_time = pd.to_datetime("2019-9-29 00:00:00")
    inputs = []
    labels = []
    while input_time <= pd.to_datetime("2019-12-29 00:00:00"):
        input = np.array(stations[id][input_time:
                                      input_time + pd.Timedelta('1 days 23 hours')])
        label = np.array(
            stations[id][input_time + pd.Timedelta('2 days'): input_time + pd.Timedelta('2 days 23 hours')])[:,
                :NUM_OUTPUTS]
        input_time += pd.Timedelta('1 days')
        input = (input - np.tile(train_mean, (len(input), 1))) / np.tile(train_std, (len(input), 1))
        inputs.append(input)
        labels.append(label)

    labels = np.array(labels)
    inputs = np.array(inputs)
    pred = multi_lstm_model.predict(inputs)
    pred = pred * np.tile(np.array(train_std[:NUM_OUTPUTS]), (len(inputs), 24, 1)) + \
           np.tile(np.array(train_mean[:NUM_OUTPUTS]), (len(inputs), 24, 1))

    smapes.append(smape(labels, pred))
    # Write data to csv
    pred_df = pd.DataFrame(columns=["Start"] + utils.OUTPUTS)
    pred_df.set_index("Start", inplace=True)
    start_time = pd.to_datetime("2019-10-1 00:00:00")
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred_df.loc[start_time] = pred[i, j]
            start_time += pd.Timedelta("1 hours")
    pred_df.to_csv(os.path.join(pred_dir, str(id) + '.csv'))

# %%

# print(np.mean(smapes))
# %%

# evaluate_avg(pred_dir='../Shenghui_centralized_model', ground_dir='../test_data')
