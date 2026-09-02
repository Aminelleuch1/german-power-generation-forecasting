import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import pytz
from pytz import timezone
from entsoe import EntsoePandasClient
from data_completer import *
from torch.optim import Adam
from darts import TimeSeries, concatenate
from darts.utils.callbacks import TFMProgressBar
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score
from sklearn.metrics import mean_squared_error
from darts.datasets import EnergyDataset
from darts import concatenate
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)


def generate_torch_kwargs():
    # run torch models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "auto",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }

def train_nbeats_model(load_train_data):
    df_nbeats = load_train_data[['Actual Load']]
    filler = MissingValuesFiller()
    scaler = Scaler()

    df_nbeats = df_nbeats[~df_nbeats.index.duplicated(keep='first')]
    df_nbeats.index = df_nbeats.index.tz_localize(None)
    df_nbeats = df_nbeats.dropna()
    df_nbeats['time'] = df_nbeats.index
    series = filler.transform(
        TimeSeries.from_dataframe(
            df_nbeats, "time", ["Actual Load"], fill_missing_dates=True, freq='H'
        )
    ).astype(np.float32)

    train, val = series.split_after(0.9)


    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    series_scaled = scaler.transform(series)
    model_name = "nbeats_run"
    model_nbeats = NBEATSModel(
        input_chunk_length=16*24*2,
        output_chunk_length=16*24,
        generic_architecture=True,
        num_stacks=10,
        num_blocks=1,
        num_layers=2,
        layer_widths=512,
        n_epochs=30,
        nr_epochs_val_period=1,
        batch_size=800,
        random_state=42,
        model_name=model_name,
        save_checkpoints=True,
        force_reset=True,
        optimizer_cls=Adam,
        **generate_torch_kwargs(),
    )
    model_nbeats.fit(train_scaled, val_series=val_scaled)
    model_nbeats = NBEATSModel.load_from_checkpoint(model_name=model_name, best=True)
    return model_nbeats



def predict_load_nbeats(client,load_df,start_date,model_nbeats):
    load_data = load_df.copy()
    load_data = complete_load(client,load_data,start_date)
    load_data = load_data[load_data.index<start_date]
    filler = MissingValuesFiller()
    scaler = Scaler()
    load_data.index = load_data.index.tz_localize(None)

    load_data['time'] = load_data.index
    series = filler.transform(
        TimeSeries.from_dataframe(
            load_data, "time", ["Actual Load"], fill_missing_dates=True, freq='H'
        )
    ).astype(np.float32)

    series_scaled = scaler.fit_transform(series)

    pp = model_nbeats.predict(
        24*16,
        series_scaled,
        verbose=True,
    )
    pp = scaler.inverse_transform(pp)
    ppv = list(pp.values())
    ppv = [item for sublist in ppv for item in sublist]
    dfpp = pp.pd_dataframe()
    dfpp.index = dfpp.index.tz_localize("UTC")
    return dfpp









def train_load_xgboost(load_train_data):
    hourly_res = load_train_data.copy()
    hourly_res['Actual Load'] = load_train_data['Actual Load'].groupby(pd.Grouper(freq='16D')).transform(lambda x: x - x.mean())
    split_index = round(0.9*len(hourly_res))
    train_df = hourly_res.iloc[:split_index]
    val_df = hourly_res.iloc[split_index:]
    # Split into features and target variable
    X_train = train_df.drop(columns=['Actual Load'])
    Y_train = train_df['Actual Load']

    X_val = val_df.drop(columns=['Actual Load'])
    Y_val = val_df['Actual Load']

    # Initialize and train the model
    model = XGBRegressor(random_state=42, objective='reg:absoluteerror', n_estimators=1500, early_stopping_rounds=100,enable_categorical=True)
    model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_val, Y_val)], verbose=100)
    return model

def predict_load(client,load_df,start_date,model_nbeats,model_xgboost,url,days=0):
    load_data = load_df.copy()
    load_data = complete_load(client,load_data,start_date)
    load_data = load_data[load_data.index<start_date]
    nbeats_df = predict_load_nbeats(client,load_data,start_date,model_nbeats)
    if(days==0):
        weather_data_forecast = complete_weather_for_load(url,end = nbeats_df.index.max(),start = nbeats_df.index.min())
    else :
        weather_data_forecast = complete_weather_for_load_test(url,end = nbeats_df.index.max(),start = nbeats_df.index.min(),days=days)
    weather_data_forecast = weather_data_forecast.dropna()
    start_date = max(nbeats_df.index.min(),weather_data_forecast.index.min())
    end_date = min(nbeats_df.index.max(),weather_data_forecast.index.max())
    weather_data_forecast = weather_data_forecast[(weather_data_forecast.index>=start_date) & (weather_data_forecast.index<=end_date)]
    nbeats_df = nbeats_df[(nbeats_df.index>=start_date) & (nbeats_df.index<=end_date)]
    data = prepare_data_for_load_xgboost_prediction(weather_data_forecast)
    xgboost_preds = model_xgboost.predict(data)
    data['load_predictions'] = xgboost_preds
    average_nbeats = np.mean(nbeats_df['Actual Load'].values)
    data['load_predictions']+=average_nbeats
    final_load_predictions = data[['load_predictions']]
    return final_load_predictions



def train_xgboost_solar_model(solar_train_data):
    features = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'dew_point_2m (°C)',
                'apparent_temperature (°C)', 'precipitation (mm)', 'rain (mm)',
                'snowfall (cm)', 'surface_pressure (hPa)', 'cloud_cover (%)',
                'cloud_cover_low (%)', 'cloud_cover_mid (%)', 'cloud_cover_high (%)',
                'wind_speed_10m (km/h)', 'year', 'month', 'day', 'day_of_week',
                'day_of_year', 'weekday', 'hour', 'week', 'season', 'sin_1', 'cos_1',
                'sin_2', 'cos_2', 'sin_3', 'cos_3']

    target = 'Solar/Capacity'
    data = solar_train_data.copy()
    split_index= round(0.9*len(data))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    
    train_data['season'] = train_data['season'].astype('category')
    test_data['season'] = test_data['season'].astype('category')

    X_train, X_val, y_train, y_val = train_test_split(train_data[features], train_data[target], test_size=0.2, random_state=42,shuffle=False)

    model = XGBRegressor(random_state=42, objective='reg:absoluteerror', n_estimators=1500, early_stopping_rounds=100, 
                        enable_categorical=True)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=100)
    return model



def predict_solar(url,model,start,end,capacity_df,days = 0):
    if(days ==0):
        weather_data_forecast = complete_weather_for_solar(url,end = end,start = start)
    else : 
        weather_data_forecast = complete_weather_for_solar_test(url,end = end,start = start,days=days)
    data = prepare_data_xgboost(weather_data_forecast)
    capacity_data = capacity_df.copy()
    capacity_data = capacity_data[(capacity_data.index>=data.index.min()) & (capacity_data.index<=data.index.max())]
    predictions = model.predict(data)
    for i in range(len(predictions)):
        predictions[i] = max(0,predictions[i])
    data['Solar_predictions'] = predictions
    data['Solar_predictions']*=capacity_df['Solar_capacity']
    data = data[['Solar_predictions']]
    return data
    




def train_xgboost_onshore_model(onshore_train_data):
    target = 'Wind_power'
    data = onshore_train_data.copy()
    split_index= round(0.9*len(data))
    train_df= data.iloc[:split_index]
    val_df = data.iloc[split_index:]
    
    train_df['season'] = train_df['season'].astype('category')
    val_df['season'] = val_df['season'].astype('category')

    X_train = train_df.drop(columns=['Wind_power'])
    Y_train = train_df['Wind_power']

    X_val = val_df.drop(columns=['Wind_power'])
    Y_val = val_df['Wind_power']

    model = XGBRegressor(random_state=42, objective='reg:absoluteerror', n_estimators=1500, early_stopping_rounds=100,enable_categorical=True)
    model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_val, Y_val)], verbose=100)
    return model



def predict_onshore(url,model,start,end,capacity_df,days = 0):
    features = ['temperature_2m (_C)_hannover', 'relative_humidity_2m (%)_hannover',
       'dewpoint_2m (_C)_hannover', 'apparent_temperature (_C)_hannover',
       'pressure_msl (hPa)_hannover', 'precipitation (mm)_hannover',
       'rain (mm)_hannover', 'snowfall (cm)_hannover',
       'cloudcover (%)_hannover', 'cloudcover_low (%)_hannover',
       'cloudcover_mid (%)_hannover', 'cloudcover_high (%)_hannover',
       'shortwave_radiation (W/m²)_hannover',
       'direct_radiation (W/m²)_hannover', 'diffuse_radiation (W/m²)_hannover',
       'windspeed_10m (km/h)_hannover', 'windspeed_100m (km/h)_hannover',
       'winddirection_10m (_)_hannover', 'winddirection_100m (_)_hannover',
       'temperature_2m (_C)_berlin', 'relative_humidity_2m (%)_berlin',
       'dewpoint_2m (_C)_berlin', 'apparent_temperature (_C)_berlin',
       'pressure_msl (hPa)_berlin', 'precipitation (mm)_berlin',
       'rain (mm)_berlin', 'snowfall (cm)_berlin', 'cloudcover (%)_berlin',
       'cloudcover_low (%)_berlin', 'cloudcover_mid (%)_berlin',
       'cloudcover_high (%)_berlin', 'shortwave_radiation (W/m²)_berlin',
       'direct_radiation (W/m²)_berlin', 'diffuse_radiation (W/m²)_berlin',
       'windspeed_10m (km/h)_berlin', 'windspeed_100m (km/h)_berlin',
       'winddirection_10m (_)_berlin', 'winddirection_100m (_)_berlin',
       'temperature_2m (_C)_bavaria', 'relative_humidity_2m (%)_bavaria',
       'dewpoint_2m (_C)_bavaria', 'apparent_temperature (_C)_bavaria',
       'pressure_msl (hPa)_bavaria', 'precipitation (mm)_bavaria',
       'rain (mm)_bavaria', 'snowfall (cm)_bavaria', 'cloudcover (%)_bavaria',
       'cloudcover_low (%)_bavaria', 'cloudcover_mid (%)_bavaria',
       'cloudcover_high (%)_bavaria', 'shortwave_radiation (W/m²)_bavaria',
       'direct_radiation (W/m²)_bavaria', 'diffuse_radiation (W/m²)_bavaria',
       'windspeed_10m (km/h)_bavaria', 'windspeed_100m (km/h)_bavaria',
       'winddirection_10m (_)_bavaria', 'winddirection_100m (_)_bavaria',
       'wind_speed^3_hannover', 'wind_speed^3_berlin', 'wind_speed^3_bavaria',
       'Wind_Onshore_capacity', 'year', 'month', 'day', 'day_of_week',
       'day_of_year', 'weekday', 'hour', 'week', 'season', 'sin_1', 'cos_1',
       'sin_2', 'cos_2', 'sin_3', 'cos_3']
    if(days == 0):
        weather_data_forecast = complete_weather_for_wind_onshore(url,end = end,start = start)
    else :
        weather_data_forecast = complete_weather_for_wind_onshore_test(url,end = end,start = start,days = days)
    data = prepare_data_xgboost(weather_data_forecast)
    capacity_data = capacity_df.copy()
    capacity_data = capacity_data[(capacity_data.index>=data.index.min()) & (capacity_data.index<=data.index.max())]
    data['Wind_Onshore_capacity'] = capacity_data['Wind_Onshore_capacity']
    data = data[features]
    predictions = model.predict(data)
    for i in range(len(predictions)):
        predictions[i] = max(0,predictions[i])
    data['onshore_predictions'] = predictions
    data['onshore_predictions']*=capacity_df['Wind_Onshore_capacity']
    data = data[['onshore_predictions']]
    df = pd.read_csv('data/generation_data.csv',index_col=0)
    df.index = pd.to_datetime(df.index,utc= True)
    df = df[(df.index>=data.index.min()) & (df.index<=data.index.max())]
    df = df[['Wind Onshore']]
    df['preds'] = data['onshore_predictions'].values
    df.plot()
    plt.show()
    return data
    






def train_xgboost_offshore_model(onshore_train_data):
    features = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'dewpoint_2m (°C)',
        'apparent_temperature (°C)', 'pressure_msl (hPa)', 'precipitation (mm)',
        'rain (mm)', 'snowfall (cm)', 'cloudcover (%)', 'cloudcover_low (%)',
        'cloudcover_mid (%)', 'cloudcover_high (%)',
        'shortwave_radiation (W/m²)', 'direct_radiation (W/m²)',
        'diffuse_radiation (W/m²)', 'windspeed_10m (km/h)',
        'windspeed_100m (km/h)', 'winddirection_10m (°)',
        'winddirection_100m (°)', 'wind_speed^3', 'OffShore_Capacity',
            'year', 'month', 'day', 'day_of_week',
        'day_of_year', 'weekday', 'hour', 'week', 'season', 'sin_1', 'cos_1',
        'sin_2', 'cos_2', 'sin_3', 'cos_3']

    target = 'Wind_power'
    data = onshore_train_data.copy()
    split_index= round(0.9*len(data))
    train_df= data.iloc[:split_index]
    val_df = data.iloc[split_index:]
    
    train_df['season'] = train_df['season'].astype('category')
    val_df['season'] = val_df['season'].astype('category')

    X_train = train_df.drop(columns=['Wind_power'])
    X_train = X_train[features]
    Y_train = train_df['Wind_power']

    X_val = val_df.drop(columns=['Wind_power'])
    Y_val = val_df['Wind_power']

    model = XGBRegressor(random_state=42, objective='reg:absoluteerror', n_estimators=1500, early_stopping_rounds=100,enable_categorical=True)
    model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_val, Y_val)], verbose=100)
    return model



def predict_offshore(url,model,start,end,capacity_df,days = 0):
    features = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'dewpoint_2m (°C)',
        'apparent_temperature (°C)', 'pressure_msl (hPa)', 'precipitation (mm)',
        'rain (mm)', 'snowfall (cm)', 'cloudcover (%)', 'cloudcover_low (%)',
        'cloudcover_mid (%)', 'cloudcover_high (%)',
        'shortwave_radiation (W/m²)', 'direct_radiation (W/m²)',
        'diffuse_radiation (W/m²)', 'windspeed_10m (km/h)',
        'windspeed_100m (km/h)', 'winddirection_10m (°)',
        'winddirection_100m (°)', 'wind_speed^3', 'OffShore_Capacity',
            'year', 'month', 'day', 'day_of_week',
        'day_of_year', 'weekday', 'hour', 'week', 'season', 'sin_1', 'cos_1',
        'sin_2', 'cos_2', 'sin_3', 'cos_3']

    target = 'Wind_power'
    if(days!=0):
        weather_data_forecast = complete_weather_for_wind_offshore_test(url,end = end,start = start,days = days)
    else :
        weather_data_forecast = complete_weather_for_wind_offshore(url,end = end,start = start)
    data = prepare_data_xgboost(weather_data_forecast)
    capacity_data = capacity_df.copy()
    capacity_data = capacity_data[(capacity_data.index>=data.index.min()) & (capacity_data.index<=data.index.max())]
    data['OffShore_Capacity'] = capacity_data['OffShore_Capacity']
    data = data[features]
    predictions = model.predict(data)
    for i in range(len(predictions)):
        predictions[i] = max(0,predictions[i])
    data['offshore_predictions'] = predictions
    data['offshore_predictions']*=capacity_df['OffShore_Capacity']
    data = data[['offshore_predictions']]
    return data
    



def predict_hydro(client,start,end):
    starth = start-pd.Timedelta(days = 3)
    endh = start-pd.Timedelta(hours = 1)
    dv = client.query_generation('DE', start = starth,end = endh, psr_type=None)
    dv.index = pd.to_datetime(dv.index,utc = True)
    # Resample the 15-minute data to hourly by summing each hour
    hourly_dv = dv.resample('H').sum()
    hourly_dv = hourly_dv[[('Hydro Run-of-river and poundage', 'Actual Aggregated')]]
    hourly_dv.columns = hourly_dv.columns.droplevel(1)
    meanh = np.mean(hourly_dv['Hydro Run-of-river and poundage'])
    time_index = pd.date_range(start=start, end=end, freq='H')

    # Create the DataFrame with 'hydro_predictions' column filled with 1
    df = pd.DataFrame(index=time_index, data={'hydro_predictions': meanh})
    return df






def predict_residuals(client,start,url,days=0):
    if(days ==0):
        load_df = pd.read_csv('data/load_data.csv',index_col=0)
        load_df.index= pd.to_datetime(load_df.index,utc =True)
        
        model_name = "nbeats_run"
        model_nbeats = NBEATSModel(
            input_chunk_length=16*24*2,
            output_chunk_length=16*24,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=2,
            layer_widths=512,
            n_epochs=30,
            nr_epochs_val_period=1,
            batch_size=800,
            random_state=42,
            model_name=model_name,
            save_checkpoints=True,
            force_reset=True,
            optimizer_cls=Adam,
            **generate_torch_kwargs(),
        )
        model_nbeats.load_weights('models/nbeats_load_model.pt', map_location="cpu")
        xgboost_model_load = xgb.XGBRegressor()
        xgboost_model_load.load_model("models/load_xgboost_model.json")
        data = predict_load(client,load_df,start,model_nbeats,xgboost_model_load,url)
        
        
        solar_model = xgb.XGBRegressor()  
        solar_model.load_model("models/solar_xgboost_model.json")
        capacities = pd.read_csv('data/capacity_data.csv',index_col = 0)
        capacities.index = pd.to_datetime(capacities.index,utc=  True)
        solar_capacity_df = create_hourly_capacity_data(capacities,"Solar","Solar_capacity",data.index.max()+pd.Timedelta(days = 3))
        df = predict_solar(url,solar_model,data.index.min(),data.index.max(),solar_capacity_df)
        data['Solar_predictions'] = df['Solar_predictions']

        onshore_model = xgb.XGBRegressor()  
        onshore_model.load_model("models/onshore_xgboost_model.json")
        onshore_capacity_df = create_hourly_capacity_data(capacities,"Wind Onshore","Wind_Onshore_capacity",data.index.max()+pd.Timedelta(days = 3))
        df = predict_onshore(url,onshore_model,data.index.min(),data.index.max(),onshore_capacity_df)
        data['onshore_predictions'] = df['onshore_predictions']
        

        offshore_model = xgb.XGBRegressor()  
        offshore_model.load_model("models/offshore_xgboost_model.json")
        offshore_capacity_df = create_hourly_capacity_data(capacities,"Wind Offshore","OffShore_Capacity",data.index.max()+pd.Timedelta(days = 3))
        df = predict_offshore(url,offshore_model,data.index.min(),data.index.max(),offshore_capacity_df)
        data['offshore_predictions'] = df['offshore_predictions']

        df= predict_hydro(client,data.index.min(),data.index.max())
        data['hydro_predictions'] = df['hydro_predictions']
        data['residuals_predictions'] = data['load_predictions']-data['Solar_predictions']-data['onshore_predictions']-data['offshore_predictions']-data['hydro_predictions']
        return data
    else :
        load_df = pd.read_csv('data/load_data.csv',index_col=0)
        load_df.index= pd.to_datetime(load_df.index,utc =True)
        
        model_name = "nbeats_run"
        model_nbeats = NBEATSModel(
            input_chunk_length=16*24*2,
            output_chunk_length=16*24,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=2,
            layer_widths=512,
            n_epochs=30,
            nr_epochs_val_period=1,
            batch_size=800,
            random_state=42,
            model_name=model_name,
            save_checkpoints=True,
            force_reset=True,
            optimizer_cls=Adam,
            **generate_torch_kwargs(),
        )
        model_nbeats.load_weights('models/nbeats_load_model.pt', map_location="cpu")
        xgboost_model_load = xgb.XGBRegressor()
        xgboost_model_load.load_model("models/load_xgboost_model.json")
        data = predict_load(client,load_df,start,model_nbeats,xgboost_model_load,url,days=days)
        
        
        solar_model = xgb.XGBRegressor()  
        solar_model.load_model("models/solar_xgboost_model.json")
        capacities = pd.read_csv('data/capacity_data.csv',index_col = 0)
        capacities.index = pd.to_datetime(capacities.index,utc=  True)
        solar_capacity_df = create_hourly_capacity_data(capacities,"Solar","Solar_capacity",data.index.max()+pd.Timedelta(days = 3))
        df = predict_solar(url,solar_model,data.index.min(),data.index.max(),solar_capacity_df,days = days)
        data['Solar_predictions'] = df['Solar_predictions']

        onshore_model = xgb.XGBRegressor()  
        onshore_model.load_model("models/onshore_xgboost_model.json")
        onshore_capacity_df = create_hourly_capacity_data(capacities,"Wind Onshore","Wind_Onshore_capacity",data.index.max()+pd.Timedelta(days = 3))
        df = predict_onshore(url,onshore_model,data.index.min(),data.index.max(),onshore_capacity_df,days = days)
        data['onshore_predictions'] = df['onshore_predictions']
        

        offshore_model = xgb.XGBRegressor()  
        offshore_model.load_model("models/offshore_xgboost_model.json")
        offshore_capacity_df = create_hourly_capacity_data(capacities,"Wind Offshore","OffShore_Capacity",data.index.max()+pd.Timedelta(days = 3))
        df = predict_offshore(url,offshore_model,data.index.min(),data.index.max(),offshore_capacity_df,days = days)
        data['offshore_predictions'] = df['offshore_predictions']

        df= predict_hydro(client,data.index.min(),data.index.max())
        data['hydro_predictions'] = df['hydro_predictions']
        print(data)
        data['residuals_predictions'] = data['load_predictions']-data['Solar_predictions']-data['onshore_predictions']-data['offshore_predictions']-data['hydro_predictions']
        return data