import os
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.spatial import distance

import utils
from preprocess import replace_missing_value

entry_point = "https://opendata-download-metobs.smhi.se"
from utils import *

measurements = {6: "Relative Humidity", 1: "temperature"}


def get_data(par, station):
    api = "api/version/1.0/parameter/%d/station/%d/period/corrected-archive/data.csv" % (par, station)
    url = os.path.join(entry_point, api)
    print(url)
    r = requests.get(url)
    return r.text


def get_line_number(phrase, text):
    for i, line in enumerate(text.splitlines(), 1):
        if phrase in line:
            return i


def find_nearest_station(par, long, lag):
    api = "api/version/1.0/parameter/%d.xml" % (par)
    r = requests.get(os.path.join(entry_point, api))
    # stations = json.loads(r.text)

    soup = BeautifulSoup(r.text, 'xml')

    station_id = 0
    min_dist = 1000000
    best_df = None
    for station in (soup.find_all("station")):
        dis = distance.euclidean([long, lag], [float(station.longitude.text), float(station.latitude.text)])
        if int(station.to.text[:4]) > 2019 and dis < 0.7:
            id = int(station.id.text)
            raw = get_data(par, id)
            if "2017-01-01;00:00:00" not in raw or "2017-01-01;01:00:00" not in raw or dis > min_dist:
                continue
            line_number = get_line_number("Datum", raw)
            df = pd.read_csv(StringIO(raw), skiprows=line_number - 1, sep=';')
            df['Datum'] = pd.to_datetime(df['Datum'] + ' ' + df['Tid (UTC)'])
            df.drop(df.columns[[3, 4, 5]], axis=1, inplace=True)

            df.drop(columns=['Tid (UTC)'], inplace=True)

            df = df.rename({'Datum': 'Start'}, axis='columns')
            df.set_index("Start", inplace=True)
            df.index = df.index + pd.Timedelta('0 days 1 hours')
            begin_index = pd.to_datetime("2014-1-1 00:00:00")
            end_index = pd.to_datetime("2019-12-31 23:00:00")
            df = df[begin_index: end_index]

            station_id = int(station.id.text)
            min_dist = dis
            best_df = df
    print(station_id)
    print(best_df.head())
    print(min_dist)

    return best_df


par, station = 1, 159880
positions = pd.read_csv('../data/station_long_lat.csv')
for id in utils.STATIONS:
    long = positions[positions['Station ID'] == id]["Longitude"].values[0]
    lag = positions[positions['Station ID'] == id]["Latitude"].values[0]
    weather_data = find_nearest_station(1, long, lag)

    print(weather_data.head())
    path = '../data/preprocessed_data/' + str(id) + '.csv'
    station_data = pd.read_csv(path)
    station_data["Start"] = pd.to_datetime(station_data["Start"])
    station_data = station_data[["Start"] + INPUTS]
    station_data.set_index("Start", inplace=True)
    station_data.update(weather_data)
    station_data = replace_missing_value(station_data)
    station_data.to_csv(path)
    print(station_data.head())
