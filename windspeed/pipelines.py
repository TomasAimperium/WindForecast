
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from windspeed.config import config


class filtering_aggregator(BaseEstimator, TransformerMixin):
    '''
    Funcion que se encarga de filtrar los valores anomalos del dataset 
    y elegir la estacion para la que se desea realizar el estudio.
    Admite como variable de entrada un dataframe y un string con el 
    numero de estacion. Ademas, se eliminan los valores ausentes.
    '''
    
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.reset_index(drop = True)
        data['Meteo Station 04 - Wind Speed(m/s)'] = data['Meteo Station 04 - Wind Speed(m/s)'].apply(lambda x : 0 if x<-1000 else x)
        data['Meteo Station 04 - Wind Direction(º)'] = data['Meteo Station 04 - Wind Direction(º)'].apply(lambda x : 0 if x<-1000 else x)
        data['Meteo Station 04 - Wind Direction Rad(rad)'] = data['Meteo Station 04 - Wind Direction Rad(rad)'].apply(lambda x : 0 if x<-10 else x)
        data['Meteo Station 04 - Atmospheric Pressure(mB)'] = data['Meteo Station 04 - Atmospheric Pressure(mB)'].apply(lambda x : 887.82 if x<500 else x)
        data['Meteo Station 04 - External Ambient Temperature(ºC)'] = data['Meteo Station 04 - External Ambient Temperature(ºC)'].apply(lambda x : 0 if x<-1000 else x)
        data['Meteo Station 04 - Humidity(%)'] = data['Meteo Station 04 - Humidity(%)'].apply(lambda x : 0 if x<-1000 else x)
        data['Meteo Station 10 - Wind Direction(º)'] = data['Meteo Station 10 - Wind Direction(º)'].apply(lambda x : 0 if x<-1000 else x)
        data['Meteo Station 10 - Wind Speed(m/s)'] = data['Meteo Station 10 - Wind Speed(m/s)'].apply(lambda x : 0 if x<-1000 else x)
        data['Meteo Station 10 - Wind Direction Rad(rad)'] = data['Meteo Station 10 - Wind Direction Rad(rad)'].apply(lambda x : 0 if x<-10 else x)
        data['Datetime'] =  pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S')
        data_agg = data.resample('5Min', on='Datetime').mean()
        data_noNa = data_agg.dropna()
        cols = []
        
        for i,j in enumerate(data_noNa.columns):

            if (self.key in j):
                cols.append(j)
        
        df4 = data_noNa[cols].reset_index(drop = True)
        features = df4.columns[df4.columns != 'Meteo Station '+ self.key +' - Wind Speed(m/s)']
        target = 'Meteo Station '+ self.key +' - Wind Speed(m/s)'
        x = df4.loc[:, features].values# Separating out the target
        y = df4.loc[:,[target]].values
        
        res = pd.concat([pd.DataFrame(x), 
                    pd.Series(y.reshape(len(y)))], 
                   axis = 1)
         
        res.columns = range(len(res.columns))
        return res
    
class savgol(BaseEstimator, TransformerMixin):
    
    '''
    Aplica a las columnas de un dataframe el filtro de Savitzky–Golay.
    Para cada columna del dataframe devuelve una version suavizada
    de la misma.
    
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        savgol_data = pd.DataFrame()
        for col in X.columns:
            savgol_data[col] = savgol_filter(X[col], 21, 1)
        
        return savgol_data
    
class Special_PCA(BaseEstimator, TransformerMixin):
    
    '''
    Algortimo de reduccion de dimensionalidad del problema:
    realiza tranformaciones algebraicas en las variables input 
    del problema y para reducir el problema a uno similar solo que
    con menos variables.
    Admite como input un dataframe y un valor numerico menor que 1
    '''
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        pc = PCA(self.key).fit_transform(X.iloc[:,:-1])
        var_y = pd.Series(X.iloc[:,-1])
        res = pd.concat([pd.DataFrame(pc),var_y],axis = 1)
        res.columns = range(res.shape[1])
        return res
    
    

class time_series_preparation_train(BaseEstimator, TransformerMixin):
    '''
    Funcion que transforma la matriz en una serie de tiempo que y
    divide esta en conjunto de entrenamiento y de test para 
    entrenar el modelo.
    '''
    
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        my_data = X.iloc[:,:].values
        
        n_future =self.key[0]
        n_past = self.key[1] 
        y_col = my_data.shape[1]-1

        data_X = []
        data_Y = []
        data_p_X = []
        data_p_Y = []

        for i in range(n_past, len(my_data) - n_future + 1):
            data_X.append(my_data[i - n_past:i, 0:my_data.shape[1]])
        #     train_Y.append(data_split[i + n_future - 1:i + n_future, 0])
            data_Y.append(my_data[i:i + n_future, y_col])
        # del data_train
        data_X, data_Y = np.array(data_X), np.array(data_Y)
        train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.33, random_state=42, shuffle=False)

        return [train_X, test_X, train_Y, test_Y]

    

class time_series_preparation_pred(BaseEstimator, TransformerMixin):

    '''
    Funcion que transforma la matriz en una serie de tiempo para realizar
    la prediccion
    '''
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
      return self
    def transform(self, X):

      n_future =self.key[0]
      n_past = self.key[1] 
      y_col = X.shape[1]-1

      pred = X.iloc[-n_past:,:].values.reshape(1,n_past,X.shape[1])
      return pred



class Special_MinMaxScaler(BaseEstimator, TransformerMixin):
    '''
    Escalado min max de las variables menos el objetivo.
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mm = MinMaxScaler().fit_transform(X.iloc[:,:-1])
        var_y = pd.Series(X.iloc[:,-1]).reset_index(drop = True)
        res = pd.concat([pd.DataFrame(mm),var_y],axis = 1)
        res.columns = range(res.shape[1])
        return res

class skew_train(BaseEstimator, TransformerMixin):
    '''
    Escalado min max de las variables menos el objetivo.
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.iloc[:,-1] = np.sqrt(X.iloc[:,-1])

        return X