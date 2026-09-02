import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import pytz
from pytz import timezone
from entsoe import EntsoePandasClient
from io import StringIO
import holidays
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from numpy.polynomial import Polynomial


countries = [
        #{"name": "Austria", "latitude": 47.3333, "longitude": 13.3333},
        #{"name": "Cyprus", "latitude": 53, "longitude": 33},
        #{"name": "Belgium", "latitude": 50.75, "longitude": 4.5},
        #{"name": "Bulgaria", "latitude": 42.6667, "longitude": 25.25},
        #{"name": "CzechRepublic", "latitude": 49.75, "longitude": 15},
        #{"name": "Denmark", "latitude": 56, "longitude": 10},
        #{"name": "Estonia", "latitude": 59, "longitude": 26},
        #{"name": "Finland", "latitude": 64, "longitude": 26},
        #{"name": "France", "latitude": 46, "longitude": 2},
        {"name": "Germany", "latitude": 51.5, "longitude": 10.5},
        #{"name": "Hungary", "latitude": 47, "longitude": 20},
       # {"name": "Ireland", "latitude": 53, "longitude": -8},
        #{"name": "Italy", "latitude": 42.8333, "longitude": 12.8333},
        #{"name": "Lithuania", "latitude": 55.4167, "longitude": 24},
        #{"name": "Luxembourg", "latitude": 49.75, "longitude": 6.1667},
       # {"name": "Netherlands", "latitude": 52, "longitude": 5.75},
        #{"name": "Poland", "latitude": 52, "longitude": 20},
       # {"name": "Portugal", "latitude": 39.6945, "longitude": -8.1306},
        #{"name": "Slovakia", "latitude": 48.6667, "longitude": 19.5},
        #{"name": "Spain", "latitude": 40, "longitude": -4},
    ]




def complete_generation(client,gen=None,end=None,start = None):
    # time_index, coal , gas, lignite,onshore, offshore , hydro , solar
    if start==None:
        start = gen.index.max()+pd.Timedelta(hours = 1)
    if(start>=end):
        return gen
    dv = client.query_generation('DE', start = start,end = end, psr_type=None)
    dv.index = pd.to_datetime(dv.index,utc = True)
    #Resample the 15-minute data to hourly by summing each hour
    df = dv.resample('H').sum()
    df.columns = df.columns.droplevel(1)
    col_counts = {}
    new_columns = []
    for col in df.columns:
        if col in col_counts:
            col_counts[col] += 1
            new_columns.append(f"{col}_{col_counts[col]}")
        else:
            col_counts[col] = 0
            new_columns.append(col)

    # Assign the new column names back to the DataFrame
    df.columns = new_columns
    df = df[["Fossil Brown coal/Lignite","Fossil Gas","Fossil Hard coal","Hydro Run-of-river and poundage","Solar","Wind Offshore","Wind Onshore"]]
    if gen is not None:
        gen = pd.concat([gen,df],axis = 0)
    else: gen = df.copy()
    gen.to_csv('data/generation_data.csv')
    return gen





def complete_load(client,load = None,end = None, start = None):
    #time_index,laod
    if start==None:
        start = load.index.max()+pd.Timedelta(hours = 1)
    if start>=end:
        return load
    dv = client.query_load('DE', start = start,end = end)
    print('load done')
    dv.index = pd.to_datetime(dv.index,utc = True)
    #Resample the 15-minute data to hourly by summing each hour
    df = dv.resample('H').sum()
    if load is not None:
        load = pd.concat([load,df],axis = 0)
    else: load = df.copy()
    load.to_csv('data/load_data.csv')
    return load




def complete_resid(client,resid_df=None,gen=None,load=None,end=None,start= None):
    #time_index, lignite, gas , coal, residuals
    if start==None:
        start = resid_df.index.max()+pd.Timedelta(hours = 1)
    if(start>=end):
        return resid_df
    gen = complete_generation(client,gen,end)
    load = complete_load(client,load,end)
    gen = gen[gen.index>=start]
    load = load[load.index>=start]
    df = gen.copy()
    df['residuals'] = load['Actual Load'] - df['Solar'] - df['Wind Onshore'] - df['Wind Offshore'] - df['Hydro Run-of-river and poundage']
    df.drop(columns = ['Hydro Run-of-river and poundage','Solar','Wind Onshore','Wind Offshore'],inplace = True)
    if resid_df is not None : 
        resid_df = pd.concat([resid_df,df],axis  = 0)
    else: resid_df = df.copy()
    resid_df.to_csv('data/residuals_data.csv')
    return resid_df



def complete_weather_for_load(url,weather_df = None,end = None,start = None):
    #time_index , relative_humidity_2m (%) , apparent_temperature (°C)  ,wind_speed_10m (km/h)  ,weather_code
    if start==None:
        start = weather_df.index.max()+pd.Timedelta(hours = 1)
    if(start>=end):
        return weather_df
    start_str = start.strftime('%Y-%m-%d')
    check = True
    tries = 5
    while(check & tries>0):
        tries-=1
        end_str = end.strftime('%Y-%m-%d')
        for country in countries:
            latitude = country["latitude"]
            longitude = country["longitude"]
            country_name = country["name"]

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_str,
                "end_date": end_str,
                "hourly": ["relative_humidity_2m", "apparent_temperature", "wind_speed_10m","weather_code"],
                "timezone": "UTC",
                "models": "best_match",
                "format": "csv"
            }
            response = requests.get(url, params=params)
            data = response.text

            if data:
                # Use StringIO to treat `data` as a file-like object
                data_io = StringIO(data)

                try:
                    # Try to read the CSV data, skipping the first two rows
                    df = pd.read_csv(data_io, skiprows=2)
                    check = False

                except pd.errors.ParserError:
                    # Decrement `current_end` by one day and try again
                    end_str = (pd.to_datetime(end_str) - timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                print(f"Weather data not available for {country_name}")
                return weather_df
    df['time'] = pd.to_datetime(df['time'],utc = True)
    df.set_index('time',inplace=  True)
    if weather_df is not None : 
        weather_df = pd.concat([weather_df,df],axis  = 0)
    else: weather_df = df.copy()
    weather_df.to_csv('data/weather_for_load.csv')
    return weather_df
        

def complete_weather_for_load_test(url,weather_df = None,end = None,start = None,days=0):
    #time_index , relative_humidity_2m (%) , apparent_temperature (°C)  ,wind_speed_10m (km/h)  ,weather_code
    if start==None:
        start = weather_df.index.max()+pd.Timedelta(hours = 1)
    if(start>=end):
        return weather_df
    start_str = start.strftime('%Y-%m-%d')
    check = True
    tries = 5
    while(check & tries>0):
        tries-=1
        end_str = end.strftime('%Y-%m-%d')
        for country in countries:
            latitude = country["latitude"]
            longitude = country["longitude"]
            country_name = country["name"]

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_str,
                "end_date": end_str,
                "hourly": [f'relative_humidity_2m_previous_day{days}', f'apparent_temperature_previous_day{days}', f'wind_speed_10m_previous_day{days}',f'weather_code_previous_day{days}'],
                "timezone": "UTC",
                "models": "best_match",
                "format": "csv"
            }
            response = requests.get(url, params=params)
            data = response.text

            if data:
                # Use StringIO to treat `data` as a file-like object
                data_io = StringIO(data)

                try:
                    # Try to read the CSV data, skipping the first two rows
                    df = pd.read_csv(data_io, skiprows=2)
                    check = False

                except pd.errors.ParserError:
                    # Decrement `current_end` by one day and try again
                    end_str = (pd.to_datetime(end_str) - timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                print(f"Weather data not available for {country_name}")
                return weather_df
    df['time'] = pd.to_datetime(df['time'],utc = True)
    df.set_index('time',inplace=  True)
    if weather_df is not None : 
        weather_df = pd.concat([weather_df,df],axis  = 0)
    else: weather_df = df.copy()
    weather_df.to_csv('data/weather_for_load.csv')
    s = f'_previous_day{days}'
    weather_df.columns = weather_df.columns.str.replace(s, '', regex=False)
    return weather_df






def prepare_data_for_load_training(load_df,weather_df,old_df=None):
    start = max(load_df.index.min(),weather_df.index.min())
    end = min(load_df.index.max(),weather_df.index.max())
    load_df = load_df[(load_df.index>=start) & (load_df.index<=end)]
    weather_df = weather_df[(weather_df.index>=start) & (weather_df.index<=end)]
    weather_df = weather_df.rename(columns = {'relative_humidity_2m (%)':'humd','apparent_temperature (°C)':'temp','wind_speed_10m (km/h)':'wnsp','weather_code (wmo code)':'cond'})
    def tstorm(x):
        # Thunderstorm codes in WMO
        if x in [95, 96, 99]:
            return 1
        return 0

    def clear(x):
        # Clear sky according to WMO code
        if x == 0:
            return 1
        return 0

    def fog(x):
        # Fog or mist codes in WMO
        if x in [45, 48]:
            return 1
        return 0

    def rain(x):
        # Rain and drizzle codes in WMO
        if x in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
            return 1
        return 0

    def cloud(x):
        # Cloudy conditions in WMO
        if x in [1, 2, 3]:
            return 1
        return 0

    def snow(x):
        # Snow or sleet codes in WMO
        if x in [71, 73, 75, 77, 85, 86]:
            return 1
        return 0

    # Create one-hot encoding
    weather_df['tstorm'] = weather_df['cond'].apply(tstorm)
    weather_df['clear'] = weather_df['cond'].apply(clear)
    weather_df['fog'] = weather_df['cond'].apply(fog)
    weather_df['rain'] = weather_df['cond'].apply(rain)
    weather_df['cloud'] = weather_df['cond'].apply(cloud)
    weather_df['snow'] = weather_df['cond'].apply(snow)
    # Drop original 'cond' column
    weather_df.drop('cond', axis=1, inplace=True)
    df_merged = load_df.merge(weather_df, how='left', left_index=True, right_index=True)
    de_holidays = holidays.Germany()
    df_merged['is_holiday'] = df_merged.index.map(lambda x: int(x.date() in de_holidays))    
    # fix missing values
    df_merged['tstorm'] = df_merged['tstorm'].bfill()
    df_merged['clear'] = df_merged['clear'].bfill()
    df_merged['fog'] = df_merged['fog'].bfill()
    df_merged['rain'] = df_merged['rain'].bfill()
    df_merged['cloud'] = df_merged['cloud'].bfill()
    df_merged['snow'] = df_merged['snow'].bfill()
    df_merged.interpolate(inplace=True)
    df_merged.bfill(inplace=True)


    df_merged['weekday'] = (df_merged.index.weekday < 5).astype(int)
    df_merged['sin_hour'] = np.sin(2*np.pi*df_merged.index.hour.values/24)
    df_merged['cos_hour'] = np.cos(2*np.pi*df_merged.index.hour.values/24)
    df_merged['humd'] = df_merged['humd']/100.0

    for year in df_merged.index.year.unique():
        # Create the date range for holidays in the current year
        holiday_start = pd.Timestamp(f'{year}-12-25').tz_localize('UTC')
        holiday_end = pd.Timestamp(f'{year + 1}-01-01').tz_localize('UTC')

        # Set 'is_holiday' to 1 for the specified date range
        df_merged.loc[holiday_start:holiday_end, 'is_holiday'] = 1
    if old_df is not None : 
        combined_df = pd.concat([old_df,df_merged],axis  = 0)
    else: combined_df = df_merged.copy()
    combined_df = combined_df.dropna()
    return combined_df
    



def prepare_data_for_load_xgboost_prediction(weather_df):
    weather_df = weather_df.rename(columns = {'relative_humidity_2m (%)':'humd','apparent_temperature (°C)':'temp','wind_speed_10m (km/h)':'wnsp','weather_code (wmo code)':'cond'})
    def tstorm(x):
        # Thunderstorm codes in WMO
        if x in [95, 96, 99]:
            return 1
        return 0

    def clear(x):
        # Clear sky according to WMO code
        if x == 0:
            return 1
        return 0

    def fog(x):
        # Fog or mist codes in WMO
        if x in [45, 48]:
            return 1
        return 0

    def rain(x):
        # Rain and drizzle codes in WMO
        if x in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
            return 1
        return 0

    def cloud(x):
        # Cloudy conditions in WMO
        if x in [1, 2, 3]:
            return 1
        return 0

    def snow(x):
        # Snow or sleet codes in WMO
        if x in [71, 73, 75, 77, 85, 86]:
            return 1
        return 0
    # Create one-hot encoding
    weather_df['tstorm'] = weather_df['cond'].apply(tstorm)
    weather_df['clear'] = weather_df['cond'].apply(clear)
    weather_df['fog'] = weather_df['cond'].apply(fog)
    weather_df['rain'] = weather_df['cond'].apply(rain)
    weather_df['cloud'] = weather_df['cond'].apply(cloud)
    weather_df['snow'] = weather_df['cond'].apply(snow)
    # Drop original 'cond' column
    weather_df.drop('cond', axis=1, inplace=True)
    df_merged = weather_df.copy()
    de_holidays = holidays.Germany()
    df_merged['is_holiday'] = df_merged.index.map(lambda x: int(x.date() in de_holidays))    
    df_merged['is_holiday'] = df_merged['is_holiday'].astype(int)
    df_merged['tstorm'] = df_merged['tstorm'].bfill()
    df_merged['clear'] = df_merged['clear'].bfill()
    df_merged['fog'] = df_merged['fog'].bfill()
    df_merged['rain'] = df_merged['rain'].bfill()
    df_merged['cloud'] = df_merged['cloud'].bfill()
    df_merged['snow'] = df_merged['snow'].bfill()
    df_merged.interpolate(inplace=True)
    df_merged.bfill(inplace=True)


    df_merged['weekday'] = (df_merged.index.weekday < 5).astype(int)
    df_merged['sin_hour'] = np.sin(2*np.pi*df_merged.index.hour.values/24)
    df_merged['cos_hour'] = np.cos(2*np.pi*df_merged.index.hour.values/24)
    df_merged['humd'] = df_merged['humd']/100.0

    for year in df_merged.index.year.unique():
        # Create the date range for holidays in the current year
        holiday_start = pd.Timestamp(f'{year}-12-25').tz_localize('UTC')
        holiday_end = pd.Timestamp(f'{year + 1}-01-01').tz_localize('UTC')

        # Set 'is_holiday' to 1 for the specified date range
        df_merged.loc[holiday_start:holiday_end, 'is_holiday'] = 1
    
    df_merged = df_merged.dropna()
    return df_merged






def complete_capacity_data(client,old_df=None,end=None,start=None):
    if((start==None) & (old_df is not None)):
        start = old_df.index.max()+pd.Timedelta(hours = 1)
    if(start==None):
        start = pd.to_datetime("2015-01-01").tz_localize(pytz.utc)
    all_data = []
    # Loop over each year in the range
    for year in range(start.year, end.year + 1):
        # Define the yearly start and end dates
        yearly_start = pd.to_datetime(f"{year}-01-01").tz_localize(pytz.utc)
        yearly_end = pd.to_datetime(f"{year}-12-31").tz_localize(pytz.utc)
        # Adjust the end date if it's beyond the overall end_date
        if yearly_end > end:
            yearly_end = end
        # Query the data for the specific year
        dv = client.query_installed_generation_capacity('DE', start=yearly_start, end=yearly_end, psr_type=None)
        dv['date'] = dv.index
        # Append the result to the list
        all_data.append(dv)

    # Concatenate all yearly data into a single DataFrame
    df = pd.concat(all_data)

    df.index = df['date']
    df = df.drop(columns = 'date')
    dv = df.copy()
    if old_df is not None : 
        dv = pd.concat([old_df,dv],axis  = 0)
    dv.index = pd.to_datetime(dv.index,utc = True) 
    dv.to_csv('data/capacity_data.csv')               
    return dv



def create_hourly_capacity_data(capacity_df,column,column_rename,end):
    capacity_data = capacity_df[[column]]
    start_date = capacity_data.index.min()
    hourly_timestamps = pd.date_range(start=start_date, end=end+pd.Timedelta(days = 20), freq='H')

    X = (capacity_data.index - start_date).total_seconds().values.reshape(-1, 1) / (3600 * 24 * 365.25)  # Convert to years
    y = capacity_data[column].values

    degree = 3  # You can change the degree as needed
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    hourly_X = (hourly_timestamps - start_date).total_seconds().values.reshape(-1, 1) / (3600 * 24 * 365.25)  # Convert to years
    hourly_X_poly = poly.transform(hourly_X)
    hourly_predictions = model.predict(hourly_X_poly)

    hourly_data = pd.DataFrame({'datetime': hourly_timestamps, column: hourly_predictions})

    hourly_data.index = hourly_data['datetime']
    hourly_data.drop(columns = ['datetime'],inplace = True)
    hourly_data= hourly_data.rename(columns = {column:column_rename})
    return hourly_data



def complete_weather_for_solar(url,weather_df = None,end = None,start = None):
    if start==None:
        start = weather_df.index.max()+pd.Timedelta(hours = 1)
    if(start>=end):
        return weather_df
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    for country in countries:
        latitude = country["latitude"]
        longitude = country["longitude"]
        country_name = country["name"]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "rain", "snowfall", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m"],
            "timezone": "UTC",
            "models": "best_match",
            "format": "csv"
        }
        response = requests.get(url, params=params)
        data = response.text

        if data:
            # Use StringIO to treat `data` as a file-like object
            data_io = StringIO(data)

            # Read the CSV data into a DataFrame, skipping the first two rows
            df = pd.read_csv(data_io, skiprows=2)


        else:
            print(f"Weather data not available for {country_name}")
            return weather_df
        df['time'] = pd.to_datetime(df['time'],utc = True)
        df.set_index('time',inplace=  True)
        if weather_df is not None : 
            weather_df = pd.concat([weather_df,df],axis  = 0)
        else: weather_df = df.copy()
        weather_df.to_csv('data/weather_for_solar.csv')
        return weather_df
    


def prepare_train_data_solar(capacity_data,weather_data,gen_data):
    capacity_df = capacity_data.copy()
    weather_df =weather_data.copy()
    gen_df = gen_data.copy()
    end = min(capacity_df.index.max(),min(weather_df.index.max(),gen_df.index.max()))
    start = max(capacity_df.index.min(),min(weather_df.index.min(),gen_df.index.min()))
    capacity_df = capacity_df[(capacity_df.index>=start) & (capacity_df.index<=end)]
    weather_df = weather_df[(weather_df.index<=end) & (weather_df.index>=start)]
    gen_df = gen_df[(gen_df.index>=start) & (gen_df.index<=end)]
    data  = weather_df.copy()
    data['Solar'] = gen_df['Solar']
    data['Solar_capacity'] = capacity_df['Solar_capacity']
    def season_determination(month):
        if month in [6,7,8,9]: #June-Sept = summer (highest need for cooling in Spain)
            return "summer"
        elif month in [1,2,12]: #Dec, Jan, Feb = winter (highest need for heating)
            return "winter"
        else:
            return "spring/fall" #all other months are spring or fall (similar)
    data['datetime'] = data.index
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
    data=Pre_Processing(data)
    data['Solar/Capacity'] = data["Solar"] / data["Solar_capacity"]
    return data



def prepare_data_xgboost(weather_data):
    data = weather_data.copy()
    def season_determination(month):
        if month in [6,7,8,9]: #June-Sept = summer (highest need for cooling in Spain)
            return "summer"
        elif month in [1,2,12]: #Dec, Jan, Feb = winter (highest need for heating)
            return "winter"
        else:
            return "spring/fall" #all other months are spring or fall (similar)
    data['datetime'] = data.index
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
    data=Pre_Processing(data)
    data['season'] = data['season'].astype('category')
    data.drop(columns = 'datetime',inplace = True)

    return data


























def complete_weather_for_wind_onshore(url,weather_df = None,end = None,start = None):
    if start==None:
        start = weather_df.index.max()+pd.Timedelta(hours = 1)
    if(start>=end):
        return weather_df
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    countries = [
        {"name": "Hanover", "latitude": 52.3759, "longitude": 9.7320},
        {"name": "Berlin", "latitude": 52.5200, "longitude": 13.4050},
        {"name": "Bavaria", "latitude": 48.7904, "longitude": 11.4979}
    ]
    dfs = []
    for country in countries:
        latitude = country["latitude"]
        longitude = country["longitude"]
        country_name = country["name"]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dewpoint_2m",
                "apparent_temperature",
                "pressure_msl",
                "precipitation",
                "rain",
                "snowfall",
                "cloudcover",
                "cloudcover_low",
                "cloudcover_mid",
                "cloudcover_high",
                "shortwave_radiation",
                "direct_radiation",
                "diffuse_radiation",
                "windspeed_10m",
                "windspeed_100m",
                "winddirection_10m",
                "winddirection_100m"
            ],
            "timezone": "UTC",
            "models": "best_match",
            "format": "csv"
        }
        response = requests.get(url, params=params)
        data = response.text

        if data:
            # Use StringIO to treat `data` as a file-like object
            data_io = StringIO(data)

            # Read the CSV data into a DataFrame, skipping the first two rows
            df = pd.read_csv(data_io, skiprows=2)
            dfs.append(df.copy())

        else:
            print(f"Weather data not available for {country_name}")
            return weather_df
        
    hanover_df = dfs[0].copy()
    berlin_df = dfs[1].copy()
    bavaria_df = dfs[2].copy()

    hanover_df['time'] = pd.to_datetime(hanover_df['time'])
    hanover_df.index = hanover_df['time']
    hanover_df.drop(columns = ['time'],inplace = True)


    berlin_df['time'] = pd.to_datetime(berlin_df['time'])
    berlin_df.index = berlin_df['time']
    berlin_df.drop(columns = ['time'],inplace = True)


    bavaria_df['time'] = pd.to_datetime(bavaria_df['time'])
    bavaria_df.index = bavaria_df['time']
    bavaria_df.drop(columns = ['time'],inplace = True)


    hanover_df = hanover_df.add_suffix('_hannover')
    berlin_df = berlin_df.add_suffix('_berlin')
    bavaria_df = bavaria_df.add_suffix('_bavaria')
    # Merge the dataframes on their indexes (which is 'time')
    merged_df = hanover_df.join(berlin_df, how='inner').join(bavaria_df, how='inner')
    merged_df = merged_df.dropna()
    
    merged_df['wind_speed^3_hannover'] = merged_df['windspeed_100m (km/h)_hannover']*merged_df['windspeed_100m (km/h)_hannover']*merged_df['windspeed_100m (km/h)_hannover']
    merged_df['wind_speed^3_berlin'] = merged_df['windspeed_100m (km/h)_berlin']*merged_df['windspeed_100m (km/h)_berlin']*merged_df['windspeed_100m (km/h)_berlin']
    merged_df['wind_speed^3_bavaria'] = merged_df['windspeed_100m (km/h)_bavaria']*merged_df['windspeed_100m (km/h)_bavaria']*merged_df['windspeed_100m (km/h)_bavaria']

    merged_df.index = merged_df.index.tz_localize('UTC')
    
    if weather_df is not None : 
        weather_df = pd.concat([weather_df,merged_df],axis  = 0)
    else: weather_df = merged_df.copy()
    weather_df.columns = [col.replace('°', '_') for col in weather_df.columns]
    weather_df.to_csv('data/weather_for_onshore.csv')
    return weather_df
    


def prepare_train_data_onshore(capacity_data,weather_data,gen_data):
    capacity_df = capacity_data.copy()
    weather_df =weather_data.copy()
    gen_df = gen_data.copy()
    end = min(capacity_df.index.max(),min(weather_df.index.max(),gen_df.index.max()))
    start = max(capacity_df.index.min(),min(weather_df.index.min(),gen_df.index.min()))
    capacity_df = capacity_df[(capacity_df.index>=start) & (capacity_df.index<=end)]
    weather_df = weather_df[(weather_df.index<=end) & (weather_df.index>=start)]
    gen_df = gen_df[(gen_df.index>=start) & (gen_df.index<=end)]
    data  = weather_df.copy()
    data['Wind Onshore'] = gen_df['Wind Onshore']
    data['Wind_Onshore_capacity'] = capacity_df['Wind_Onshore_capacity']
    def season_determination(month):
        if month in [6,7,8,9]: #June-Sept = summer (highest need for cooling in Spain)
            return "summer"
        elif month in [1,2,12]: #Dec, Jan, Feb = winter (highest need for heating)
            return "winter"
        else:
            return "spring/fall" #all other months are spring or fall (similar)
    data['datetime'] = data.index
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
    data=Pre_Processing(data)
    data['Wind_power'] = data["Wind Onshore"] / data["Wind_Onshore_capacity"]
    data=data.drop(columns=["Wind Onshore",'datetime'])
    data.columns = [col.replace('°', '_') for col in data.columns]
    return data



def prepare_data_xgboost_onshore(weather_data):
    data = weather_data.copy()
    def season_determination(month):
        if month in [6,7,8,9]: #June-Sept = summer (highest need for cooling in Spain)
            return "summer"
        elif month in [1,2,12]: #Dec, Jan, Feb = winter (highest need for heating)
            return "winter"
        else:
            return "spring/fall" #all other months are spring or fall (similar)
    data['datetime'] = data.index
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
    data=Pre_Processing(data)
    data['season'] = data['season'].astype('category')
    data.drop(columns = 'datetime',inplace = True)

    return data


















def complete_weather_for_wind_offshore(url,weather_data = None,end = None,start = None):
    if start==None:
        start = weather_df.index.max()+pd.Timedelta(hours = 1)
    if(start>=end):
        return weather_df
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    countries = [
        {"name": "North_sea", "latitude": 52.0433334, "longitude": 7.5},
        {"name": "Baltic_sea", "latitude": 54.79789, "longitude": 12.83099}
    ]
    dfs = []
    for country in countries:
        latitude = country["latitude"]
        longitude = country["longitude"]
        country_name = country["name"]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dewpoint_2m",
                "apparent_temperature",
                "pressure_msl",
                "precipitation",
                "rain",
                "snowfall",
                "cloudcover",
                "cloudcover_low",
                "cloudcover_mid",
                "cloudcover_high",
                "shortwave_radiation",
                "direct_radiation",
                "diffuse_radiation",
                "windspeed_10m",
                "windspeed_100m",
                "winddirection_10m",
                "winddirection_100m"
            ],
            "timezone": "UTC",
            "models": "best_match",
            "format": "csv"
        }
        response = requests.get(url, params=params)
        data = response.text

        if data:
            # Use StringIO to treat `data` as a file-like object
            data_io = StringIO(data)

            # Read the CSV data into a DataFrame, skipping the first two rows
            df = pd.read_csv(data_io, skiprows=2)
            dfs.append(df.copy())

        else:
            print(f"Weather data not available for {country_name}")
            return weather_df
        
    weather_baltic_sea=dfs[1].copy()
    weather_north_sea=dfs[0].copy()
    columns_to_rename = {
        'winddirection_10m (%)': 'winddirection_10m (°)',
        'winddirection_100m (%)': 'winddirection_100m (°)'
    }

    # Rename only if the column exists
    weather_north_sea = weather_north_sea.rename(
        columns={k: v for k, v in columns_to_rename.items() if k in weather_north_sea.columns}
    )
    numeric_columns = weather_north_sea.select_dtypes(include='number').columns

    weather = 0.84 * weather_north_sea[numeric_columns] + 0.16 * weather_baltic_sea[numeric_columns]

    non_numeric_columns = weather_north_sea.select_dtypes(exclude='number').columns
    for column in non_numeric_columns:
        weather[column] = weather_north_sea[column]
    weather_df = weather.copy()
    weather_df['time'] = pd.to_datetime(weather_df['time'],utc = True)
    weather_df.index = weather_df['time']
    weather_df.drop(columns = ['time'],inplace = True)
    weather_df["wind_speed^3"]=weather_df['windspeed_100m (km/h)']**3
    weather_df = weather_df.dropna()
    if weather_data is not None : 
        weather_df = pd.concat([weather_data,weather_df],axis  = 0)
    weather_df.to_csv("data/weather_data_offshore.csv")
    return weather_df
    


def prepare_train_data_offshore(capacity_data,weather_data,gen_data):
    capacity_df = capacity_data.copy()
    weather_df =weather_data.copy()
    gen_df = gen_data.copy()
    end = min(capacity_df.index.max(),min(weather_df.index.max(),gen_df.index.max()))
    start = max(capacity_df.index.min(),min(weather_df.index.min(),gen_df.index.min()))
    capacity_df = capacity_df[(capacity_df.index>=start) & (capacity_df.index<=end)]
    weather_df = weather_df[(weather_df.index<=end) & (weather_df.index>=start)]
    gen_df = gen_df[(gen_df.index>=start) & (gen_df.index<=end)]
    data  = weather_df.copy()
    data['Wind Offshore'] = gen_df['Wind Offshore']
    data['OffShore_Capacity'] = capacity_df['OffShore_Capacity']
    def season_determination(month):
        if month in [6,7,8,9]: #June-Sept = summer (highest need for cooling in Spain)
            return "summer"
        elif month in [1,2,12]: #Dec, Jan, Feb = winter (highest need for heating)
            return "winter"
        else:
            return "spring/fall" #all other months are spring or fall (similar)
    data['datetime'] = data.index
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
    data=Pre_Processing(data)
    data['Wind_power'] = data["Wind Offshore"] / data["OffShore_Capacity"]
    data=data.drop(columns=["Wind Offshore",'datetime'])
    return data



def prepare_data_xgboost_offshore(weather_data):
    data = weather_data.copy()
    def season_determination(month):
        if month in [6,7,8,9]: #June-Sept = summer (highest need for cooling in Spain)
            return "summer"
        elif month in [1,2,12]: #Dec, Jan, Feb = winter (highest need for heating)
            return "winter"
        else:
            return "spring/fall" #all other months are spring or fall (similar)
    data['datetime'] = data.index
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
    data=Pre_Processing(data)
    data['season'] = data['season'].astype('category')
    data.drop(columns = 'datetime',inplace = True)

    return data









def complete_weather_for_solar_test(url,weather_df = None,end = None,start = None,days = 0):
    for country in countries:
        latitude = country["latitude"]
        longitude = country["longitude"]
        country_name = country["name"]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [f"temperature_2m_previous_day{days}", f"relative_humidity_2m_previous_day{days}", f"dew_point_2m_previous_day{days}", f"apparent_temperature_previous_day{days}", f"precipitation_previous_day{days}", f"rain_previous_day{days}", f"snowfall_previous_day{days}", f"surface_pressure_previous_day{days}", f"cloud_cover_previous_day{days}", f"cloud_cover_low_previous_day{days}", f"cloud_cover_mid_previous_day{days}", f"cloud_cover_high_previous_day{days}", f"wind_speed_10m_previous_day{days}"],
            "timezone": "UTC",
            "format": "csv",
            "past_days": 92,
	        "forecast_days": 16
        }
        response = requests.get(url, params=params)
        data = response.text
        if data:
            # Use StringIO to treat `data` as a file-like object
            data_io = StringIO(data)

            # Read the CSV data into a DataFrame, skipping the first two rows
            df = pd.read_csv(data_io, skiprows=2)


        else:
            print(f"Weather data not available for {country_name}")
            return weather_df
        df['time'] = pd.to_datetime(df['time'],utc = True)
        df.set_index('time',inplace=  True)
        if weather_df is not None : 
            weather_df = pd.concat([weather_df,df],axis  = 0)
        else: weather_df = df.copy()
        weather_df.to_csv('data/weather_forecast_solar.csv')
        s = f'_previous_day{days}'
        weather_df.columns = weather_df.columns.str.replace(s, '', regex=False)
        return weather_df








def complete_weather_for_wind_onshore_test(url,weather_df = None,end = None,start = None,days = 0):
    if start==None:
        start = weather_df.index.max()+pd.Timedelta(hours = 1)
    if(start>=end):
        return weather_df
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    countries = [
        {"name": "Hanover", "latitude": 52.3759, "longitude": 9.7320},
        {"name": "Berlin", "latitude": 52.5200, "longitude": 13.4050},
        {"name": "Bavaria", "latitude": 48.7904, "longitude": 11.4979}
    ]
    dfs = []
    for country in countries:
        latitude = country["latitude"]
        longitude = country["longitude"]
        country_name = country["name"]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": [
                f"temperature_2m_previous_day{days}",
                f"relative_humidity_2m_previous_day{days}",
                f"dewpoint_2m_previous_day{days}",
                f"apparent_temperature_previous_day{days}",
                f"pressure_msl_previous_day{days}",
                f"precipitation_previous_day{days}",
                f"rain_previous_day{days}",
                f"snowfall_previous_day{days}",
                f"cloudcover_previous_day{days}",
                f"cloudcover_low_previous_day{days}",
                f"cloudcover_mid_previous_day{days}",
                f"cloudcover_high_previous_day{days}",
                f"shortwave_radiation_previous_day{days}",
                f"direct_radiation_previous_day{days}",
                f"diffuse_radiation_previous_day{days}",
                f"windspeed_10m_previous_day{days}",
                f"windspeed_100m_previous_day{days}",
                f"winddirection_10m_previous_day{days}",
                f"winddirection_100m_previous_day{days}"
            ],
            "timezone": "UTC",
            "models": "best_match",
            "format": "csv"
        }
        response = requests.get(url, params=params)
        data = response.text

        if data:
            # Use StringIO to treat `data` as a file-like object
            data_io = StringIO(data)

            # Read the CSV data into a DataFrame, skipping the first two rows
            df = pd.read_csv(data_io, skiprows=2)
            s = f'_previous_day{days}'
            df.columns = df.columns.str.replace(s, '', regex=False)
            dfs.append(df.copy())

        else:
            print(f"Weather data not available for {country_name}")
            return weather_df
        
    hanover_df = dfs[0].copy()
    berlin_df = dfs[1].copy()
    bavaria_df = dfs[2].copy()

    hanover_df['time'] = pd.to_datetime(hanover_df['time'])
    hanover_df.index = hanover_df['time']
    hanover_df.drop(columns = ['time'],inplace = True)

    berlin_df['time'] = pd.to_datetime(berlin_df['time'])
    berlin_df.index = berlin_df['time']
    berlin_df.drop(columns = ['time'],inplace = True)

    bavaria_df['time'] = pd.to_datetime(bavaria_df['time'])
    bavaria_df.index = bavaria_df['time']
    bavaria_df.drop(columns = ['time'],inplace = True)


    hanover_df = hanover_df.add_suffix('_hannover')
    berlin_df = berlin_df.add_suffix('_berlin')
    bavaria_df = bavaria_df.add_suffix('_bavaria')
    # Merge the dataframes on their indexes (which is 'time')
    merged_df = hanover_df.join(berlin_df, how='inner').join(bavaria_df, how='inner')

    merged_df.fillna(0, inplace=True)
    
    merged_df['wind_speed^3_hannover'] = merged_df['windspeed_100m (km/h)_hannover']*merged_df['windspeed_100m (km/h)_hannover']*merged_df['windspeed_100m (km/h)_hannover']
    merged_df['wind_speed^3_berlin'] = merged_df['windspeed_100m (km/h)_berlin']*merged_df['windspeed_100m (km/h)_berlin']*merged_df['windspeed_100m (km/h)_berlin']
    merged_df['wind_speed^3_bavaria'] = merged_df['windspeed_100m (km/h)_bavaria']*merged_df['windspeed_100m (km/h)_bavaria']*merged_df['windspeed_100m (km/h)_bavaria']

    merged_df.index = merged_df.index.tz_localize('UTC')
    print(merged_df)
    if weather_df is not None : 
        weather_df = pd.concat([weather_df,merged_df],axis  = 0)
    else: weather_df = merged_df.copy()
    weather_df.columns = [col.replace('°', '_') for col in weather_df.columns]
    weather_df.to_csv('data/weather_for_onshore.csv')
    print(weather_df)
    weather_data_forecast = complete_weather_for_wind_onshore(url,end = end,start = start)
    df = weather_df[['windspeed_10m (km/h)_hannover']]
    print(weather_df)
    print(df)
    df['preds'] = weather_data_forecast['windspeed_10m (km/h)_hannover'].values
    print(df)
    df.plot()
    plt.show()
    print('done')
    return weather_df
    


















def complete_weather_for_wind_offshore_test(url,weather_data = None,end = None,start = None,days = 0):
    if start==None:
        start = weather_df.index.max()+pd.Timedelta(hours = 1)
    if(start>=end):
        return weather_df
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    countries = [
        {"name": "North_sea", "latitude": 52.0433334, "longitude": 7.5},
        {"name": "Baltic_sea", "latitude": 54.79789, "longitude": 12.83099}
    ]
    dfs = []
    for country in countries:
        latitude = country["latitude"]
        longitude = country["longitude"]
        country_name = country["name"]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": [
                f"temperature_2m_previous_day{days}",
                f"relative_humidity_2m_previous_day{days}",
                f"dewpoint_2m_previous_day{days}",
                f"apparent_temperature_previous_day{days}",
                f"pressure_msl_previous_day{days}",
                f"precipitation_previous_day{days}",
                f"rain_previous_day{days}",
                f"snowfall_previous_day{days}",
                f"cloudcover_previous_day{days}",
                f"cloudcover_low_previous_day{days}",
                f"cloudcover_mid_previous_day{days}",
                f"cloudcover_high_previous_day{days}",
                f"shortwave_radiation_previous_day{days}",
                f"direct_radiation_previous_day{days}",
                f"diffuse_radiation_previous_day{days}",
                f"windspeed_10m_previous_day{days}",
                f"windspeed_100m_previous_day{days}",
                f"winddirection_10m_previous_day{days}",
                f"winddirection_100m_previous_day{days}"
            ],
            "timezone": "UTC",
            "models": "best_match",
            "format": "csv"
        }
        response = requests.get(url, params=params)
        data = response.text

        if data:
            # Use StringIO to treat `data` as a file-like object
            data_io = StringIO(data)

            # Read the CSV data into a DataFrame, skipping the first two rows
            df = pd.read_csv(data_io, skiprows=2)
            s = f'_previous_day{days}'
            df.columns = df.columns.str.replace(s, '', regex=False)
            dfs.append(df.copy())

        else:
            print(f"Weather data not available for {country_name}")
            return weather_df
        
    weather_baltic_sea=dfs[1].copy()
    weather_north_sea=dfs[0].copy()
    columns_to_rename = {
        'winddirection_10m (%)': 'winddirection_10m (°)',
        'winddirection_100m (%)': 'winddirection_100m (°)'
    }

    # Rename only if the column exists
    weather_north_sea = weather_north_sea.rename(
        columns={k: v for k, v in columns_to_rename.items() if k in weather_north_sea.columns}
    )
    numeric_columns = weather_north_sea.select_dtypes(include='number').columns

    weather = 0.84 * weather_north_sea[numeric_columns] + 0.16 * weather_baltic_sea[numeric_columns]

    non_numeric_columns = weather_north_sea.select_dtypes(exclude='number').columns
    for column in non_numeric_columns:
        weather[column] = weather_north_sea[column]
    weather_df = weather.copy()
    weather_df['time'] = pd.to_datetime(weather_df['time'],utc = True)
    weather_df.index = weather_df['time']
    weather_df.drop(columns = ['time'],inplace = True)
    weather_df["wind_speed^3"]=weather_df['windspeed_100m (km/h)']**3
    weather_df = weather_df.fillna(0)
    if weather_data is not None : 
        weather_df = pd.concat([weather_data,weather_df],axis  = 0)
    weather_df.to_csv("data/weather_data_offshore.csv")
    return weather_df
