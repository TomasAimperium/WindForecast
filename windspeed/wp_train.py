import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy.signal import savgol_filter
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import load_model

from windspeed.config import config
from windspeed import pipelines



def train_w():

    print(config.DATASET_DIR + "/train.csv")
    all_data = pd.read_csv(config.DATASET_DIR + "/train.csv",sep=";",decimal=",")
    trunc_days = config.DAYS
    trunc_mins = trunc_days*24*60


    data = all_data[0:trunc_mins]


    data_prep =  Pipeline([
	    ('filtering_aggregator', pipelines.filtering_aggregator(key = config.STATION)),
	    ('skew',pipelines.skew_train()),
	    ('scale',pipelines.Special_MinMaxScaler()),
	    ('savgol',pipelines.savgol()),
	    ('Special_PCA',pipelines.Special_PCA(key = config.PCA_N)),
	    ('scale2',pipelines.Special_MinMaxScaler())
    ])

    train_pipe = Pipeline([
	    ('data_pred', data_prep),
	    ('time_series_preparation_train',pipelines.time_series_preparation_train(key = config.DATA_RANGE))

    ])

    pred_pipe =  Pipeline([
	    ('data_pred',data_prep),
	    ('time_series_preparation_pred',pipelines.time_series_preparation_pred(key = config.DATA_RANGE))                       
    ]).fit(data)


    tr = train_pipe.fit_transform(data)
    trainx = tr[0].shape
    trainy = tr[2].shape

    #entrenamiento del modelo.
    neurons = config.NEURONS
    keras.backend.clear_session()
    lstm_model = Sequential()
    lstm_model.add(Input(shape=[trainx[-2], trainx[-1]]))
    lstm_model.add(Dense(10))
    lstm_model.add(LSTM(neurons, activation='tanh', input_shape=(trainx[1], trainx[2]), return_sequences=False))
    lstm_model.add(Dense(30))
    lstm_model.add(Dense(trainy[1]))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(tr[0],tr[2],epochs = config.EPOCHS, batch_size = config.BATCH,validation_data=(tr[1], tr[3]),verbose=1)

    #guardado del modelo
    lstm_model.save(config.TRAINED_MODEL_DIR+"/"+config.MODEL_NAME+".h5")
    return"hola train"




def pred_w():

    all_data = pd.read_csv(config.DATASET_DIR + "/train.csv",sep=";",decimal=",")
    trunc_days = config.DAYS
    trunc_mins = trunc_days*24*60
    data = all_data[0:trunc_mins]

    data_prep =  Pipeline([
        ('filtering_aggregator', pipelines.filtering_aggregator(key = config.STATION)),
        ('skew',pipelines.skew_train()),
        ('scale',pipelines.Special_MinMaxScaler()),
        ('savgol',pipelines.savgol()),
        ('Special_PCA',pipelines.Special_PCA(key = config.PCA_N)),
        ('scale2',pipelines.Special_MinMaxScaler())
    ])

    pred_pipe =  Pipeline([
        ('data_pred',data_prep),
        ('time_series_preparation_pred',pipelines.time_series_preparation_pred(key = config.DATA_RANGE))                       
    ]).fit(data)

    data = all_data[trunc_mins:trunc_mins + 1000]
    my_data = pred_pipe.transform(data)
    lstm_model = load_model(config.TRAINED_MODEL_DIR+"/"+config.MODEL_NAME+".h5")
    lstm_model.predict(my_data)**2

    return lstm_model.predict(my_data)**2