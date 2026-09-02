import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime,timezone
import pytz

from entsoe import EntsoePandasClient
from data_completer import *
from model_training import *
import xgboost as xgb

from supabase import create_client, Client
from Import_Data_Supabase import *
from Preprocessing_Data import *
from Training_Inference import *


#open meteo archive data url
base_url_archive = "https://archive-api.open-meteo.com/v1/archive"
base_url_forecast = "https://api.open-meteo.com/v1/forecast"
base_url_historical_forecasts = "https://historical-forecast-api.open-meteo.com/v1/forecast"
historical_runs_url = "https://previous-runs-api.open-meteo.com/v1/forecast"


#Connecting to entsoe API
# Credentials come from the environment — never hard-code them here.
# See .env.example for the full list of variables and how to obtain them.
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY")
if not ENTSOE_API_KEY:
    raise RuntimeError(
        "ENTSOE_API_KEY is not set. Request a free token from the ENTSO-E "
        "Transparency Platform (see README) and export it before running:\n"
        "    export ENTSOE_API_KEY='your-token'"
    )
client = EntsoePandasClient(api_key=ENTSOE_API_KEY)


"""
#Complete generation
df = pd.read_csv('data/generation_data.csv',index_col = 0)
df.index = pd.to_datetime(df.index,utc = True)
df = complete_generation(client,df,pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True))
df.to_csv('data/generation_data.csv')

#Complete load
load_df = pd.read_csv('data/load_data.csv')
load_df.index= pd.to_datetime(load_df.index,utc = True)
load_df = complete_load(client,load_df,pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True))
load_df.to_csv('data/load_data.csv')


#Complete residuals
resid_df = pd.read_csv('data/residuals_data.csv')
resid_df.index = pd.to_datetime(resid_df.index,utc = True)
resid_df = complete_resid(client,resid_df,df,load_df,pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True))
resid_df.to_csv('data/residuals_data.csv')

#Complete weather for load
weather_load= pd.read_csv('data/weather_for_load.csv',index_col= 0)
weather_load.index = pd.to_datetime(weather_load.index,utc = True)
weather_load = complete_weather_for_load(base_url_archive,weather_load,pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True))
weather_load.to_csv('data/weather_for_load.csv')

weather_load= pd.read_csv('data/weather_for_load.csv',index_col= 0)
weather_load.index = pd.to_datetime(weather_load.index,utc = True)



load_df = pd.read_csv('data/load_data.csv',index_col = 0)
load_df.index= pd.to_datetime(load_df.index,utc = True)


load_train_data = prepare_data_for_load_training(load_df,weather_load)
load_train_data.to_csv('data/load_train_data.csv')

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
load_xgboost_model = xgb.XGBRegressor()  
load_xgboost_model.load_model("models/load_xgboost_model.json")

predictions = predict_load(client,load_df,weather_load,pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True),model_nbeats,load_xgboost_model,base_url_forecast)


cap_data = complete_capacity_data(client=client,end = pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True))
cap_data.to_csv('data/capacity_data.csv')"""
"""cap_data = pd.read_csv('data/capacity_data.csv',index_col = 0)
cap_data.index = pd.to_datetime(cap_data.index,utc= True)


hourly_solar_cap_data = create_hourly_capacity_data(cap_data,'Solar','Solar_capacity',pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True))
hourly_solar_cap_data.to_csv('data/hourly_solar_capacity_data.csv')


weather_solar_df = pd.read_csv('data/weather_for_solar.csv',index_col = 0)
weather_solar_df.index = pd.to_datetime(weather_solar_df.index,utc = True)

gen_df = pd.read_csv('data/generation_data.csv',index_col = 0)
gen_df.index = pd.to_datetime(gen_df.index,utc = True)

capacity_df = pd.read_csv('data/hourly_solar_capacity_data.csv',index_col = 0)
capacity_df.index = pd.to_datetime(capacity_df.index,utc = True)

data = prepare_train_data_solar(capacity_df,weather_solar_df,gen_df)

model = xgb.XGBRegressor()
model.load_model("models/solar_xgboost_model.json")
predictions = predict_solar(base_url_forecast,model,pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True),pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True)+pd.Timedelta(days = 16),capacity_df)
predictions.plot()
plt.show()
"""
"""
hourly_onshore_cap_data = pd.read_csv('data/hourly_onshore_capacity_data.csv',index_col = 0)
hourly_onshore_cap_data.index = pd.to_datetime(hourly_onshore_cap_data.index,utc = True)


#onshore_weather_data = complete_weather_for_wind_onshore(base_url_archive,start = pd.to_datetime('2016-01-01 00:00:00+00:00',utc = True),end = pd.to_datetime('2024-10-11 22:00:00+00:00',utc =True))

onshore_weather_data = pd.read_csv('data/weather_for_onshore.csv',index_col = 0)
onshore_weather_data.index = pd.to_datetime(onshore_weather_data.index,utc = True)


gen_df = pd.read_csv('data/generation_data.csv',index_col = 0)
gen_df.index = pd.to_datetime(gen_df.index,utc = True)

onshore_train_data = prepare_train_data_onshore(hourly_onshore_cap_data,onshore_weather_data,gen_df)
onshore_train_data = pd.read_csv('data/onshore_train_data.csv',index_col = 0)
onshore_train_data.index=  pd.to_datetime(onshore_train_data.index,utc = True)


model = train_xgboost_onshore_model(onshore_train_data)
model.save_model("models/onshore_xgboost_model.json")


onshore_xgboost_model = xgb.XGBRegressor()  
onshore_xgboost_model.load_model("models/onshore_xgboost_model.json")


hourly_onshore_cap_data = pd.read_csv('data/hourly_onshore_capacity_data.csv',index_col = 0)
hourly_onshore_cap_data.index = pd.to_datetime(hourly_onshore_cap_data.index,utc = True)
data = predict_onshore(base_url_forecast,onshore_xgboost_model,pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True),pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True)+pd.Timedelta(days = 16),hourly_onshore_cap_data)


hourly_offshore_cap_data = pd.read_csv('data/hourly_offshore_capacity_data.csv',index_col = 0)
hourly_offshore_cap_data.index= pd.to_datetime(hourly_offshore_cap_data.index,utc = True)

weather_data_offshore = pd.read_csv('data/weather_data_offshore.csv',index_col = 0)
weather_data_offshore.index= pd.to_datetime(weather_data_offshore.index,utc = True)

gen_df = pd.read_csv('data/generation_data.csv',index_col = 0)
gen_df.index = pd.to_datetime(gen_df.index,utc = True)

offshore_train_data = pd.read_csv('data/offshore_train_data.csv',index_col = 0)
offshore_train_data.index = pd.to_datetime(offshore_train_data.index,utc= True)

model = train_xgboost_offshore_model(offshore_train_data)
model.save_model("models/offshore_xgboost_model.json")


offshore_xgboost_model = xgb.XGBRegressor()  
offshore_xgboost_model.load_model("models/offshore_xgboost_model.json")


hourly_offshore_cap_data = pd.read_csv('data/hourly_offshore_capacity_data.csv',index_col = 0)
hourly_offshore_cap_data.index = pd.to_datetime(hourly_offshore_cap_data.index,utc = True)
data = predict_offshore(base_url_forecast,offshore_xgboost_model,pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True),pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True)+pd.Timedelta(days = 16),hourly_offshore_cap_data)


data = predict_hydro(client,pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True),pd.to_datetime("2024-10-04 23:00:00+00:00",utc = True)+pd.Timedelta(days = 16))
"""































#Testing now







"""

utc_date = pd.to_datetime(datetime.now(timezone('UTC')).strftime("%Y-%m-%d"),utc=  True)

s = int(input("do you want predictions or past 0(predictions) 1(past) : "))
max_date = utc_date-pd.Timedelta(days = 16)
if(s==0):
   data = predict_residuals(client,utc_date,base_url_forecast)
   data_resid = data[['residuals_predictions']]
else :
    date = input('date (yyyy-mm--dd): ')
    date = pd.to_datetime(date,utc=  True)
    if(date>max_date):
        print('invalid_date to test')
    else :
        data = predict_residuals(client,date,base_url_historical_forecasts)
        data_resid = data[['residuals_predictions']]
        gen_df= pd.read_csv('data/generation_data.csv',index_col = 0)
        gen_df.index= pd.to_datetime(gen_df.index,utc= True)

        load_df= pd.read_csv('data/load_data.csv',index_col = 0)
        load_df.index= pd.to_datetime(load_df.index,utc= True)


        residuals_data = pd.read_csv('data/residuals_data.csv',index_col = 0)
        residuals_data.index = pd.to_datetime(residuals_data.index,utc = True)
        residuals_data = complete_resid(client,residuals_data,gen_df,load_df,data.index.max()+pd.Timedelta(hours = 2))
        res_data = residuals_data[(residuals_data.index>=data.index.min()) & (residuals_data.index<=data.index.max())]
        data_resid['residuals'] = res_data['residuals']


data_resid.plot()
data.to_csv('data/residual_predictions.csv')
plt.show()

"""







"""
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
data_frames=fetch_data_by_date(SUPABASE_URL, SUPABASE_KEY)
"""

#Preprocess Gas Data
gas_prices=pd.read_csv("data/Gas_TTF_rows.csv") 
gas_prices = gas_futures_preprocessing_for_model_train(gas_prices)

#Preprocess Coal Data
hourly_mean_df_Coal = treat_Coal_data("data/Coal_rows.csv")



#Preprocess EUA PRICES Data
hourly_mean_df_EUA = treat_EUA_data("data/EUA_rows.csv")

best_model_full = xgb.XGBRegressor()
best_model_full.load_model('models/best_xgboost_model_full_data (5).json')



utc_date = pd.to_datetime(datetime.now(timezone('UTC')).strftime("%Y-%m-%d"),utc=  True)
s = int(input("do you want predictions or past 0(predictions) 1(past) : "))
max_date = utc_date-pd.Timedelta(days = 16)
if(s==0):
    data = predict_residuals(client,utc_date,base_url_forecast)
    preds_residuals=data[["residuals_predictions"]]
    preds_residuals.rename(columns = {"residuals_predictions":"residuals"},inplace= True)
    preds_residuals.index = preds_residuals.index.tz_localize(None)
    preds_residuals['datetime'] = preds_residuals.index
    preds_residuals['datetime'] = pd.to_datetime(preds_residuals['datetime']).dt.tz_localize(None)
    gas_prices=pd.read_csv("data/Gas_TTF_rows.csv") 
    gas_prices = gas_futures_preprocessing_for_model_train(gas_prices)

    #Preprocess Coal Data
    hourly_mean_df_Coal = treat_Coal_data("data/Coal_rows.csv")



    #Preprocess EUA PRICES Data
    hourly_mean_df_EUA = treat_EUA_data("data/EUA_rows.csv")

    test=pd.merge(preds_residuals,hourly_mean_df_Coal,on="datetime",how="inner")
    test=pd.merge(test,gas_prices,on="datetime",how="inner")
    test=pd.merge(test,hourly_mean_df_EUA,on="datetime",how="inner")
    # Create datetime and fourrier features
    def Pre_Processing(df):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df["year"] = df.datetime.dt.year
        df["month"] = df.datetime.dt.month
        df["day"] = df.datetime.dt.day
        df["day_of_week"] = df.datetime.dt.day_of_week
        df["day_of_year"] = df.datetime.dt.day_of_year
        df["weekday"] = df.datetime.dt.weekday
        df['hour'] = df['datetime'].dt.hour
        df['week'] = df['datetime'].dt.isocalendar().week
        df['season'] = df.month.apply(season_determination)
        num_features=1
        frequencies = [(n / 24) for n in range(1, num_features + 1)]
        for n in range(1, num_features + 1):
            df[f'sin_{n}'] = np.sin( np.pi * frequencies[n - 1] * df.hour)
            df[f'cos_{n}'] = np.cos( np.pi * frequencies[n - 1] * df.hour)
        frequencies = [(n / 12) for n in range(1, num_features + 1)]
        for n in range(2, num_features + 2):
            df[f'sin_{n}'] = np.sin( np.pi * frequencies[n - 2] * df.month)
            df[f'cos_{n}'] = np.cos( np.pi * frequencies[n - 2] * df.month)
        frequencies = [(n / 366) for n in range(1, num_features + 1)]
        for n in range(3, num_features + 3):
            df[f'sin_{n}'] = np.sin( np.pi * frequencies[n - 3] * df.day_of_year)
            df[f'cos_{n}'] = np.cos( np.pi * frequencies[n - 3] * df.day_of_year)
        return df

    test=Pre_Processing(test)
    test["residuals/Prices"]=test["residuals"]/test["High_month_ago"]
    test.rename(columns={"Open_month_ago":"Open_month_ago_TTF",'High_month_ago':'High_month_ago_TTF','Close_month_ago':'Close_month_ago_TTF','mean_price_close':'mean_price_close_TTF', 'mean_price_open':"'mean_price_open'_TTF", 'mean_price_high':'mean_price_high_TTF'},inplace=True)
    test['Gas_Cost'] = 0.5 * test['mean_price_open_EUA'] + test["'mean_price_open'_TTF"]

    test['Coal_Cost'] = 1.1 * test['mean_price_open_EUA'] + test['mean_price_open_Coal']

    test['Cost_Difference'] = test['Coal_Cost'] - test['Gas_Cost']

    test['Is_Coal_Cost_Higher'] = test['Coal_Cost'] > test['Gas_Cost']



    test = pd.get_dummies(test, columns=["season"])


    test.set_index('datetime',inplace = True)
    X_test =test.copy()
    # Set 'season_winter' to zero if it doesn't exist in X_test columns
    if 'season_winter' not in X_test.columns:
        X_test['season_winter'] = 0

    # Set 'season_spring/fall' to zero if it doesn't exist in X_test columns
    if 'season_spring/fall' not in X_test.columns:
        X_test['season_spring/fall'] = 0

    # Set 'season_summer' to zero if it doesn't exist in X_test columns
    if 'season_summer' not in X_test.columns:
        X_test['season_summer'] = 0

    features =['Open_month_ago_TTF', 'High_month_ago_TTF', 'Close_month_ago_TTF',
        'mean_price_close_TTF', "'mean_price_open'_TTF", 'mean_price_high_TTF',
        'Open_16days_ago', 'High_16days_ago', 'Close_16days_ago', 'month',
        'residuals', 'year', 'day', 'day_of_week', 'day_of_year', 'weekday',
        'hour', 'week', 'sin_1', 'cos_1', 'sin_2', 'cos_2', 'sin_3', 'cos_3',
        'residuals/Prices', 'mean_price_close_Coal', 'mean_price_open_Coal',
        'mean_price_high_Coal', 'mean_price_close_EUA', 'mean_price_open_EUA',
        'mean_price_high_EUA', 'Gas_Cost', 'Coal_Cost', 'Cost_Difference',
        'Is_Coal_Cost_Higher', 'season_spring/fall', 'season_summer',
        'season_winter']
    X_test = X_test[features]


    predictions_flat =best_model_full.predict(X_test)

    # Reshape predictions back to (n_samples, 3)
    predictions = predictions_flat.reshape(-1, 3)
    predictions = predictions

    # Convert predictions to DataFrame for easier handling
    predictions_df = pd.DataFrame(predictions, columns=["Fossil Gas_Pred", "Fossil Hard coal_Pred", "Fossil Brown coal/Lignite_Pred"])
    predictions_df["datetime"] = X_test.index.values


    #predictions_df = pd.read_csv('data/predictions.csv',index_col = 0)
    predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'],utc= True)
    predictions_df.set_index('datetime',inplace= True)
    # CO2 emission factors (Kg of CO2/MWh)
    CO2_emission_factors = {
        "Fossil Gas": 185,
        "Fossil Hard coal": 920,
        "Fossil Brown coal/Lignite": 1183
    }

    comparison_df = predictions_df.copy()
    comparison_df["Total_CO2_Pred"] = (
        comparison_df["Fossil Gas_Pred"] * CO2_emission_factors["Fossil Gas"] +
        comparison_df["Fossil Hard coal_Pred"] * CO2_emission_factors["Fossil Hard coal"] +
        comparison_df["Fossil Brown coal/Lignite_Pred"] * CO2_emission_factors["Fossil Brown coal/Lignite"]
    )

    comparison_df.to_csv('data/comparison.csv')
    # Plotting actual vs predicted CO2 emissions
    plt.figure(figsize=(12, 6))

    plt.plot(comparison_df.index, comparison_df["Total_CO2_Pred"], label='Predicted Total CO2 Emissions', color='green', linewidth=2.5)

    plt.title('Predicted Total CO2 Emissions')
    plt.xlabel('Datetime')
    plt.ylabel('CO2 Emissions (Kg)')
    plt.legend()

    plt.tight_layout()
    plt.show()

else :
    date = input('date (yyyy-mm-dd): ')
    utc_date = pd.to_datetime(date,utc = True)
    utcc_date = utc_date
    if(utc_date>max_date):
        print('invalid_date to test')
    else :
        for days in range(0,8):    
            utc_date = utcc_date
            data = predict_residuals(client,utc_date,historical_runs_url,days = days)
            preds_residuals=data[["residuals_predictions"]]
            if(days==0):
                resid = pd.read_csv('data/residuals_data.csv',index_col=0)
                resid.index = pd.to_datetime(resid.index,utc=  True)
                resid = resid[['residuals']]
                resid = resid[(resid.index>=preds_residuals.index.min()) & (resid.index<=preds_residuals.index.max())]
                resid[f'preds {days}'] = preds_residuals['residuals_predictions']
                resid.plot()
                plt.show()
            else :
                resid[f'preds {days}'] = preds_residuals['residuals_predictions']
                print(resid)
                resid.plot()
                plt.show()
            preds_residuals.rename(columns = {"residuals_predictions":"residuals"},inplace= True)
            preds_residuals.index = preds_residuals.index.tz_localize(None)
            preds_residuals['datetime'] = preds_residuals.index
            preds_residuals['datetime'] = pd.to_datetime(preds_residuals['datetime']).dt.tz_localize(None)
            test=pd.merge(preds_residuals,hourly_mean_df_Coal,on="datetime",how="inner")
            test=pd.merge(test,gas_prices,on="datetime",how="inner")
            test=pd.merge(test,hourly_mean_df_EUA,on="datetime",how="inner")
            print("test",test)
            # Create datetime and fourrier features
            def Pre_Processing(df):
                df['datetime'] = pd.to_datetime(df['datetime'])
                df["year"] = df.datetime.dt.year
                df["month"] = df.datetime.dt.month
                df["day"] = df.datetime.dt.day
                df["day_of_week"] = df.datetime.dt.day_of_week
                df["day_of_year"] = df.datetime.dt.day_of_year
                df["weekday"] = df.datetime.dt.weekday
                df['hour'] = df['datetime'].dt.hour
                df['week'] = df['datetime'].dt.isocalendar().week
                df['season'] = df.month.apply(season_determination)
                num_features=1
                frequencies = [(n / 24) for n in range(1, num_features + 1)]
                for n in range(1, num_features + 1):
                    df[f'sin_{n}'] = np.sin( np.pi * frequencies[n - 1] * df.hour)
                    df[f'cos_{n}'] = np.cos( np.pi * frequencies[n - 1] * df.hour)
                frequencies = [(n / 12) for n in range(1, num_features + 1)]
                for n in range(2, num_features + 2):
                    df[f'sin_{n}'] = np.sin( np.pi * frequencies[n - 2] * df.month)
                    df[f'cos_{n}'] = np.cos( np.pi * frequencies[n - 2] * df.month)
                frequencies = [(n / 366) for n in range(1, num_features + 1)]
                for n in range(3, num_features + 3):
                    df[f'sin_{n}'] = np.sin( np.pi * frequencies[n - 3] * df.day_of_year)
                    df[f'cos_{n}'] = np.cos( np.pi * frequencies[n - 3] * df.day_of_year)
                return df

            test=Pre_Processing(test)
            test["residuals/Prices"]=test["residuals"]/test["High_month_ago"]
            test.rename(columns={"Open_month_ago":"Open_month_ago_TTF",'High_month_ago':'High_month_ago_TTF','Close_month_ago':'Close_month_ago_TTF','mean_price_close':'mean_price_close_TTF', 'mean_price_open':"'mean_price_open'_TTF", 'mean_price_high':'mean_price_high_TTF'},inplace=True)
            test['Gas_Cost'] = 0.5 * test['mean_price_open_EUA'] + test["'mean_price_open'_TTF"]

            test['Coal_Cost'] = 1.1 * test['mean_price_open_EUA'] + test['mean_price_open_Coal']

            test['Cost_Difference'] = test['Coal_Cost'] - test['Gas_Cost']

            test['Is_Coal_Cost_Higher'] = test['Coal_Cost'] > test['Gas_Cost']


            test = pd.get_dummies(test, columns=["season"])


            test.set_index('datetime',inplace = True)
            X_test =test.copy()
            # Set 'season_winter' to zero if it doesn't exist in X_test columns
            if 'season_winter' not in X_test.columns:
                X_test['season_winter'] = 0

            # Set 'season_spring/fall' to zero if it doesn't exist in X_test columns
            if 'season_spring/fall' not in X_test.columns:
                X_test['season_spring/fall'] = 0

            # Set 'season_summer' to zero if it doesn't exist in X_test columns
            if 'season_summer' not in X_test.columns:
                X_test['season_summer'] = 0

            features =['Open_month_ago_TTF', 'High_month_ago_TTF', 'Close_month_ago_TTF',
                'mean_price_close_TTF', "'mean_price_open'_TTF", 'mean_price_high_TTF',
                'Open_16days_ago', 'High_16days_ago', 'Close_16days_ago', 'month',
                'residuals', 'year', 'day', 'day_of_week', 'day_of_year', 'weekday',
                'hour', 'week', 'sin_1', 'cos_1', 'sin_2', 'cos_2', 'sin_3', 'cos_3',
                'residuals/Prices', 'mean_price_close_Coal', 'mean_price_open_Coal',
                'mean_price_high_Coal', 'mean_price_close_EUA', 'mean_price_open_EUA',
                'mean_price_high_EUA', 'Gas_Cost', 'Coal_Cost', 'Cost_Difference',
                'Is_Coal_Cost_Higher', 'season_spring/fall', 'season_summer',
                'season_winter']
            X_test = X_test[features]


            predictions_flat =best_model_full.predict(X_test)

            # Reshape predictions back to (n_samples, 3)
            predictions = predictions_flat.reshape(-1, 3)

            # Convert predictions to DataFrame for easier handling
            predictions_df = pd.DataFrame(predictions, columns=["Fossil Gas_Pred", "Fossil Hard coal_Pred", "Fossil Brown coal/Lignite_Pred"])
            predictions_df["datetime"] = X_test.index.values


            #predictions_df = pd.read_csv('data/predictions.csv',index_col = 0)
            predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'],utc= True)
            predictions_df.set_index('datetime',inplace= True)
            
            gen = pd.read_csv('data/generation_data.csv',index_col=0)
            gen.index= pd.to_datetime(gen.index,utc= True)
            utc_date = pd.to_datetime(datetime.now(timezone('UTC')).strftime("%Y-%m-%d"),utc=  True)
            gen = complete_generation(client,gen,utc_date)
            print("gen",gen)
            print("pred",predictions_df)
            gen = gen[(gen.index>=predictions_df.index.min()) & (gen.index<=predictions_df.index.max())]
            predictions_df['Fossil Gas'] = gen['Fossil Gas'].values
            predictions_df['Fossil Hard coal'] = gen['Fossil Hard coal'].values
            predictions_df['Fossil Brown coal/Lignite'] = gen['Fossil Brown coal/Lignite'].values











            comparison_df = predictions_df.copy()


            # Multiply predictions back by the 'sum' feature to get actual scale
            # predictions_df["Fossil Gas_Pred"] *= test_data["sum"].values
            # predictions_df["Fossil Hard coal_Pred"] *= test_data["sum"].values
            # predictions_df["Fossil Brown coal/Lignite_Pred"] *= test_data["sum"].values

            # Combine actual and predicted data for visualization
            #comparison_df = pd.concat([preds_residuals[[ "Fossil Gas", "Fossil Hard coal", "Fossil Brown coal/Lignite"]].reset_index(drop=True),
             #                       predictions_df[["Fossil Gas_Pred", "Fossil Hard coal_Pred", "Fossil Brown coal/Lignite_Pred"]].reset_index(drop=True)], axis=1)
            

            # Plotting the actual vs predicted values for each target
            plt.figure(figsize=(12, 8))

            # Fossil Gas
            plt.subplot(3, 1, 1)
            plt.plot(comparison_df.index, comparison_df["Fossil Gas"], label='Actual Fossil Gas', color='green', linewidth=2.5)
            plt.plot(comparison_df.index, comparison_df["Fossil Gas_Pred"], label='Predicted Fossil Gas', color='green', linestyle='dashed', linewidth=2.5)
            plt.title('Actual vs Predicted Fossil Gas')
            plt.legend()

            # Fossil Hard Coal
            plt.subplot(3, 1, 2)
            plt.plot(comparison_df.index, comparison_df["Fossil Hard coal"], label='Actual Fossil Hard coal', color='green', linewidth=2.5)
            plt.plot(comparison_df.index, comparison_df["Fossil Hard coal_Pred"], label='Predicted Fossil Hard coal', color='green', linestyle='dashed', linewidth=2.5)
            plt.title('Actual vs Predicted Fossil Hard Coal')
            plt.legend()

            # Fossil Brown Coal/Lignite
            plt.subplot(3, 1, 3)
            plt.plot(comparison_df.index, comparison_df["Fossil Brown coal/Lignite"], label='Actual Fossil Brown coal/Lignite', color='green', linewidth=2.5)
            plt.plot(comparison_df.index, comparison_df["Fossil Brown coal/Lignite_Pred"], label='Predicted Fossil Brown coal/Lignite', color='green', linestyle='dashed', linewidth=2.5)
            plt.title('Actual vs Predicted Fossil Brown Coal/Lignite')
            plt.legend()

            plt.tight_layout()
            plt.show()

            from sklearn.metrics import mean_absolute_error, mean_squared_error

            # Function to calculate RMSE
            def rmse(y_true, y_pred):
                return np.sqrt(mean_squared_error(y_true, y_pred))

            # Calculate MAE and RMSE for each output
            mae_fossil_gas = mean_absolute_error(comparison_df["Fossil Gas"], comparison_df["Fossil Gas_Pred"])
            rmse_fossil_gas = rmse(comparison_df["Fossil Gas"], comparison_df["Fossil Gas_Pred"])

            mae_fossil_hard_coal = mean_absolute_error(comparison_df["Fossil Hard coal"], comparison_df["Fossil Hard coal_Pred"])
            rmse_fossil_hard_coal = rmse(comparison_df["Fossil Hard coal"], comparison_df["Fossil Hard coal_Pred"])

            mae_fossil_brown_coal = mean_absolute_error(comparison_df["Fossil Brown coal/Lignite"], comparison_df["Fossil Brown coal/Lignite_Pred"])
            rmse_fossil_brown_coal = rmse(comparison_df["Fossil Brown coal/Lignite"], comparison_df["Fossil Brown coal/Lignite_Pred"])

            # Display the results
            print(f"MAE for Fossil Gas: {mae_fossil_gas}")
            print(f"RMSE for Fossil Gas: {rmse_fossil_gas}\n")

            print(f"MAE for Fossil Hard Coal: {mae_fossil_hard_coal}")
            print(f"RMSE for Fossil Hard Coal: {rmse_fossil_hard_coal}\n")

            print(f"MAE for Fossil Brown Coal/Lignite: {mae_fossil_brown_coal}")
            print(f"RMSE for Fossil Brown Coal/Lignite: {rmse_fossil_brown_coal}")


            # CO2 emission factors (Kg of CO2/MWh)
            CO2_emission_factors = {
                "Fossil Gas": 185,
                "Fossil Hard coal": 920,
                "Fossil Brown coal/Lignite": 1183
            }

            # Calculate total CO2 emissions (actual and predicted)
            comparison_df["Total_CO2_Actual"] = (
                comparison_df["Fossil Gas"] * CO2_emission_factors["Fossil Gas"] +
                comparison_df["Fossil Hard coal"] * CO2_emission_factors["Fossil Hard coal"] +
                comparison_df["Fossil Brown coal/Lignite"] * CO2_emission_factors["Fossil Brown coal/Lignite"]
            )

            comparison_df["Total_CO2_Pred"] = (
                comparison_df["Fossil Gas_Pred"] * CO2_emission_factors["Fossil Gas"] +
                comparison_df["Fossil Hard coal_Pred"] * CO2_emission_factors["Fossil Hard coal"] +
                comparison_df["Fossil Brown coal/Lignite_Pred"] * CO2_emission_factors["Fossil Brown coal/Lignite"]
            )

            comparison_df.to_csv(f'data/comparison{days}_day_before.csv')
            # Plotting actual vs predicted CO2 emissions
            plt.figure(figsize=(12, 6))

            # Plotting total CO2 emissions
            plt.plot(comparison_df.index, comparison_df["Total_CO2_Actual"], label='Actual Total CO2 Emissions', color='green', linewidth=2.5)
            plt.plot(comparison_df.index, comparison_df["Total_CO2_Pred"], label='Predicted Total CO2 Emissions', color='green', linestyle='dashed', linewidth=2.5)

            plt.title(f'Actual vs Predicted Total CO2 Emissions {days}')
            plt.xlabel('Datetime')
            plt.ylabel('CO2 Emissions (Kg)')
            plt.legend()

            plt.tight_layout()
            plt.show()

            mae_fossil_hard_coal = mean_absolute_error(comparison_df["Total_CO2_Actual"], comparison_df["Total_CO2_Pred"])
            rmse_fossil_hard_coal = rmse(comparison_df["Total_CO2_Actual"], comparison_df["Total_CO2_Pred"])
            print(f"MAE for CO2 Emissions: {mae_fossil_hard_coal}")
            print(f"RMSE for CO2 Emissions: {rmse_fossil_hard_coal}")




"""
utc_date = pd.to_datetime(datetime.now(timezone('UTC')).strftime("%Y-%m-%d"),utc=  True)
utc_date = pd.to_datetime('2024-10-01',utc = True)
data = predict_residuals(client,utc_date,base_url_forecast)
preds_residuals=data[["residuals_predictions"]]
preds_residuals.rename(columns = {"residuals_predictions":"residuals"},inplace= True)
preds_residuals.index = preds_residuals.index.tz_localize(None)
preds_residuals['datetime'] = preds_residuals.index
preds_residuals['datetime'] = pd.to_datetime(preds_residuals['datetime']).dt.tz_localize(None)
test=pd.merge(preds_residuals,hourly_mean_df_Coal,on="datetime",how="inner")
test=pd.merge(test,gas_prices,on="datetime",how="inner")
test=pd.merge(test,hourly_mean_df_EUA,on="datetime",how="inner")
# Create datetime and fourrier features
def Pre_Processing(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df["year"] = df.datetime.dt.year
    df["month"] = df.datetime.dt.month
    df["day"] = df.datetime.dt.day
    df["day_of_week"] = df.datetime.dt.day_of_week
    df["day_of_year"] = df.datetime.dt.day_of_year
    df["weekday"] = df.datetime.dt.weekday
    df['hour'] = df['datetime'].dt.hour
    df['week'] = df['datetime'].dt.isocalendar().week
    df['season'] = df.month.apply(season_determination)
    num_features=1
    frequencies = [(n / 24) for n in range(1, num_features + 1)]
    for n in range(1, num_features + 1):
        df[f'sin_{n}'] = np.sin( np.pi * frequencies[n - 1] * df.hour)
        df[f'cos_{n}'] = np.cos( np.pi * frequencies[n - 1] * df.hour)
    frequencies = [(n / 12) for n in range(1, num_features + 1)]
    for n in range(2, num_features + 2):
        df[f'sin_{n}'] = np.sin( np.pi * frequencies[n - 2] * df.month)
        df[f'cos_{n}'] = np.cos( np.pi * frequencies[n - 2] * df.month)
    frequencies = [(n / 366) for n in range(1, num_features + 1)]
    for n in range(3, num_features + 3):
        df[f'sin_{n}'] = np.sin( np.pi * frequencies[n - 3] * df.day_of_year)
        df[f'cos_{n}'] = np.cos( np.pi * frequencies[n - 3] * df.day_of_year)
    return df

test=Pre_Processing(test)
test["residuals/Prices"]=test["residuals"]/test["High_month_ago"]
test.rename(columns={"Open_month_ago":"Open_month_ago_TTF",'High_month_ago':'High_month_ago_TTF','Close_month_ago':'Close_month_ago_TTF','mean_price_close':'mean_price_close_TTF', 'mean_price_open':"'mean_price_open'_TTF", 'mean_price_high':'mean_price_high_TTF'},inplace=True)
test['Gas_Cost'] = 0.5 * test['mean_price_open_EUA'] + test["'mean_price_open'_TTF"]

test['Coal_Cost'] = 1.1 * test['mean_price_open_EUA'] + test['mean_price_open_Coal']

test['Cost_Difference'] = test['Coal_Cost'] - test['Gas_Cost']

test['Is_Coal_Cost_Higher'] = test['Coal_Cost'] > test['Gas_Cost']



test = pd.get_dummies(test, columns=["season"])


test.set_index('datetime',inplace = True)
X_test =test.copy()
X_test["season_winter"]=0
X_test["season_spring/fall"]=0
X_test['season_summer'] = 0

features =['Open_month_ago_TTF', 'High_month_ago_TTF', 'Close_month_ago_TTF',
       'mean_price_close_TTF', "'mean_price_open'_TTF", 'mean_price_high_TTF',
       'Open_16days_ago', 'High_16days_ago', 'Close_16days_ago', 'month',
       'residuals', 'year', 'day', 'day_of_week', 'day_of_year', 'weekday',
       'hour', 'week', 'sin_1', 'cos_1', 'sin_2', 'cos_2', 'sin_3', 'cos_3',
       'residuals/Prices', 'mean_price_close_Coal', 'mean_price_open_Coal',
       'mean_price_high_Coal', 'mean_price_close_EUA', 'mean_price_open_EUA',
       'mean_price_high_EUA', 'Gas_Cost', 'Coal_Cost', 'Cost_Difference',
       'Is_Coal_Cost_Higher', 'season_spring/fall', 'season_summer',
       'season_winter']
X_test = X_test[features]


predictions_flat =best_model_full.predict(X_test)

# Reshape predictions back to (n_samples, 3)
predictions = predictions_flat.reshape(-1, 3)
predictions = predictions

# Convert predictions to DataFrame for easier handling
predictions_df = pd.DataFrame(predictions, columns=["Fossil Gas_Pred", "Fossil Hard coal_Pred", "Fossil Brown coal/Lignite_Pred"])
predictions_df["datetime"] = X_test.index.values


#predictions_df = pd.read_csv('data/predictions.csv',index_col = 0)
predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'],utc= True)
predictions_df.set_index('datetime',inplace= True)

gen = pd.read_csv('data/generation_data.csv',index_col=0)
gen.index= pd.to_datetime(gen.index,utc= True)
utc_date = pd.to_datetime(datetime.now(timezone('UTC')).strftime("%Y-%m-%d"),utc=  True)
gen = complete_generation(client,gen,utc_date)
gen = gen[(gen.index>=predictions_df.index.min()) & (gen.index<=predictions_df.index.max())]
predictions_df['Fossil Gas'] = gen['Fossil Gas']
predictions_df['Fossil Hard coal'] = gen['Fossil Hard coal']
predictions_df['Fossil Brown coal/Lignite'] = gen['Fossil Brown coal/Lignite']













comparison_df = predictions_df.copy()



# Multiply predictions back by the 'sum' feature to get actual scale
# predictions_df["Fossil Gas_Pred"] *= test_data["sum"].values
# predictions_df["Fossil Hard coal_Pred"] *= test_data["sum"].values
# predictions_df["Fossil Brown coal/Lignite_Pred"] *= test_data["sum"].values

# Combine actual and predicted data for visualization
comparison_df = pd.concat([preds_residuals[[ "Fossil Gas", "Fossil Hard coal", "Fossil Brown coal/Lignite"]].reset_index(drop=True),
                           predictions_df[["Fossil Gas_Pred", "Fossil Hard coal_Pred", "Fossil Brown coal/Lignite_Pred"]].reset_index(drop=True)], axis=1)


# Plotting the actual vs predicted values for each target
plt.figure(figsize=(12, 8))

# Fossil Gas
plt.subplot(3, 1, 1)
plt.plot(comparison_df.index, comparison_df["Fossil Gas"], label='Actual Fossil Gas', color='blue')
plt.plot(comparison_df.index, comparison_df["Fossil Gas_Pred"], label='Predicted Fossil Gas', color='red', linestyle='dashed')
plt.title('Actual vs Predicted Fossil Gas')
plt.legend()

# Fossil Hard Coal
plt.subplot(3, 1, 2)
plt.plot(comparison_df.index, comparison_df["Fossil Hard coal"], label='Actual Fossil Hard coal', color='blue')
plt.plot(comparison_df.index, comparison_df["Fossil Hard coal_Pred"], label='Predicted Fossil Hard coal', color='red', linestyle='dashed')
plt.title('Actual vs Predicted Fossil Hard Coal')
plt.legend()

# Fossil Brown Coal/Lignite
plt.subplot(3, 1, 3)
plt.plot(comparison_df.index, comparison_df["Fossil Brown coal/Lignite"], label='Actual Fossil Brown coal/Lignite', color='blue')
plt.plot(comparison_df.index, comparison_df["Fossil Brown coal/Lignite_Pred"], label='Predicted Fossil Brown coal/Lignite', color='red', linestyle='dashed')
plt.title('Actual vs Predicted Fossil Brown Coal/Lignite')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Calculate MAE and RMSE for each output
mae_fossil_gas = mean_absolute_error(comparison_df["Fossil Gas"], comparison_df["Fossil Gas_Pred"])
rmse_fossil_gas = rmse(comparison_df["Fossil Gas"], comparison_df["Fossil Gas_Pred"])

mae_fossil_hard_coal = mean_absolute_error(comparison_df["Fossil Hard coal"], comparison_df["Fossil Hard coal_Pred"])
rmse_fossil_hard_coal = rmse(comparison_df["Fossil Hard coal"], comparison_df["Fossil Hard coal_Pred"])

mae_fossil_brown_coal = mean_absolute_error(comparison_df["Fossil Brown coal/Lignite"], comparison_df["Fossil Brown coal/Lignite_Pred"])
rmse_fossil_brown_coal = rmse(comparison_df["Fossil Brown coal/Lignite"], comparison_df["Fossil Brown coal/Lignite_Pred"])

# Display the results
print(f"MAE for Fossil Gas: {mae_fossil_gas}")
print(f"RMSE for Fossil Gas: {rmse_fossil_gas}\n")

print(f"MAE for Fossil Hard Coal: {mae_fossil_hard_coal}")
print(f"RMSE for Fossil Hard Coal: {rmse_fossil_hard_coal}\n")

print(f"MAE for Fossil Brown Coal/Lignite: {mae_fossil_brown_coal}")
print(f"RMSE for Fossil Brown Coal/Lignite: {rmse_fossil_brown_coal}")


# CO2 emission factors (Kg of CO2/MWh)
CO2_emission_factors = {
    "Fossil Gas": 185,
    "Fossil Hard coal": 920,
    "Fossil Brown coal/Lignite": 1183
}

# Calculate total CO2 emissions (actual and predicted)
comparison_df["Total_CO2_Actual"] = (
    comparison_df["Fossil Gas"] * CO2_emission_factors["Fossil Gas"] +
    comparison_df["Fossil Hard coal"] * CO2_emission_factors["Fossil Hard coal"] +
    comparison_df["Fossil Brown coal/Lignite"] * CO2_emission_factors["Fossil Brown coal/Lignite"]
)

comparison_df["Total_CO2_Pred"] = (
    comparison_df["Fossil Gas_Pred"] * CO2_emission_factors["Fossil Gas"] +
    comparison_df["Fossil Hard coal_Pred"] * CO2_emission_factors["Fossil Hard coal"] +
    comparison_df["Fossil Brown coal/Lignite_Pred"] * CO2_emission_factors["Fossil Brown coal/Lignite"]
)

comparison_df.to_csv('data/comparison.csv')
# Plotting actual vs predicted CO2 emissions
plt.figure(figsize=(12, 6))

# Plotting total CO2 emissions
plt.plot(comparison_df.index, comparison_df["Total_CO2_Actual"], label='Actual Total CO2 Emissions', color='blue')
plt.plot(comparison_df.index, comparison_df["Total_CO2_Pred"], label='Predicted Total CO2 Emissions', color='red', linestyle='dashed')

plt.title('Actual vs Predicted Total CO2 Emissions')
plt.xlabel('Datetime')
plt.ylabel('CO2 Emissions (Kg)')
plt.legend()

plt.tight_layout()
plt.show()

mae_fossil_hard_coal = mean_absolute_error(comparison_df["Total_CO2_Actual"], comparison_df["Total_CO2_Pred"])
rmse_fossil_hard_coal = rmse(comparison_df["Total_CO2_Actual"], comparison_df["Total_CO2_Pred"])
print(f"MAE for CO2 Emissions: {mae_fossil_hard_coal}")
print(f"RMSE for CO2 Emissions: {rmse_fossil_hard_coal}")

"""