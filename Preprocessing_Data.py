import pandas as pd 
import numpy as np 
import sys
import warnings
import itertools
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
import calendar
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import pandas as pd

# Create new features and modify for gas future prices
def gas_futures_preprocessing_for_model_train(gas_prices):
    # Convert date column to datetime
    gas_prices["Date"] = pd.to_datetime(gas_prices["Date"])
   
    # Month code to number mapping
    month_codes = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,
        'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10,
        'X': 11, 'Z': 12
    }
   
    # Extract expiration date from symbol
    def extract_expiration_date(symbol):
        month_code = symbol[-3]
        year_str = '20' + symbol[-2:]
        month = month_codes[month_code]
        return pd.to_datetime(f'{year_str}-{month:02d}-01')
   
    gas_prices['expiration_date'] = gas_prices['Symbol'].apply(extract_expiration_date)
   
    # Calculate mean prices in a 3-month period before expiration date
    mean_prices = []
    for exp_date in gas_prices['expiration_date'].unique():
        start_date = exp_date - pd.DateOffset(months=4)
        end_date = exp_date - pd.DateOffset(months=1)
       
        mask = (gas_prices['Date'] >= start_date) & (gas_prices['Date'] <= end_date)
        mean_prices.append({
            'expiration_date': exp_date,
            'mean_price_close': gas_prices.loc[mask, 'Close'].mean(),
            'mean_price_open': gas_prices.loc[mask, 'Open'].mean(),
            'mean_price_high': gas_prices.loc[mask, 'High'].mean()
        })
   
    mean_prices_df = pd.DataFrame(mean_prices)
   # mean_prices_df = mean_prices_df[mean_prices_df["expiration_date"] <= "2025-01-01"]
   
    # Extract expiration information from symbol
    def extract_expiration_info(symbol):
        month_symbol = symbol[2]
        year = '20' + symbol[3:]
        month = month_codes.get(month_symbol)
        return month, int(year)
   
    gas_prices['expiration_month'], gas_prices['expiration_year'] = zip(*gas_prices['Symbol'].apply(extract_expiration_info))
    gas_prices['expiration_date'] = pd.to_datetime(
        gas_prices['expiration_year'].astype(str) + '-' + gas_prices['expiration_month'].astype(str) + '-01'
    )
   
    # Calculate one month before expiration data
    def get_one_month_before_expiration(expiration_year, expiration_month):
        expiration_date = pd.to_datetime(f'{expiration_year}-{expiration_month}-01')
        return expiration_date - pd.DateOffset(months=1)
   
    data = []
    unique_symbols = gas_prices['Symbol'].unique()
   
    for symbol in unique_symbols:
        symbol_data = gas_prices[gas_prices['Symbol'] == symbol]
        expiration_year = symbol_data['expiration_year'].iloc[0]
        expiration_month = symbol_data['expiration_month'].iloc[0]
        one_month_before_expiration = get_one_month_before_expiration(expiration_year, expiration_month)
 #       print(symbol_data['expiration_date'].values[0])
       # symbol_data['Date'] = pd.to_datetime(symbol_data['Date']).dt.strftime("%Y-%m-%d %H:%M:%S")
       # print(one_month_before_expiration,formatted_expdate)
        if one_month_before_expiration in symbol_data['Date'].values:
            data.append({
                'Symbol': symbol,
                'Expiration Year': expiration_year,
                'Expiration Month': expiration_month,
                'Date': one_month_before_expiration,
                'Open': symbol_data[symbol_data['Date'] == one_month_before_expiration]['Open'].values[0],
                'High': symbol_data[symbol_data['Date'] == one_month_before_expiration]['High'].values[0],
                'Close': symbol_data[symbol_data['Date'] == one_month_before_expiration]['Close'].values[0]            })
   
    prices_df = pd.DataFrame(data)
    prices_df.rename(columns={"Open": "Open_month_ago", "High": "High_month_ago", "Close": "Close_month_ago"}, inplace=True)
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    prices_df["Date"]= prices_df["Date"]+pd.DateOffset(months=1)
    prices_df.set_index('Date', inplace=True)
   
    # Resample to hourly
    hourly_df = prices_df.resample('H').ffill()
    mean_prices_df['expiration_date'] = pd.to_datetime(mean_prices_df['expiration_date'])
    mean_prices_df.set_index('expiration_date', inplace=True)
    hourly_mean_df = mean_prices_df.resample('H').ffill()
   
    # Collect prices 16 days before expiration
    def get_days_before_expiration(expiration_year, expiration_month, days_before):
        expiration_date = pd.to_datetime(f'{expiration_year}-{expiration_month}-01')
        return expiration_date - pd.DateOffset(days=days_before)

    data_16_days_before = []
    filtered_data = gas_prices
   
    for symbol in unique_symbols:
        symbol_data = filtered_data[filtered_data['Symbol'] == symbol]
        expiration_year = symbol_data['expiration_year'].iloc[0]
        expiration_month = symbol_data['expiration_month'].iloc[0]
       
        for days_before in range(16, 0, -1):
            date_before_expiration = get_days_before_expiration(expiration_year, expiration_month, days_before)
            if date_before_expiration in symbol_data['Date'].values:
                data_16_days_before.append({
                    'Symbol': symbol,
                    'Expiration Year': expiration_year,
                    'Expiration Month': expiration_month,
                    'Date': date_before_expiration,
                    'Open': symbol_data[symbol_data['Date'] == date_before_expiration]['Open'].values[0],
                    'High': symbol_data[symbol_data['Date'] == date_before_expiration]['High'].values[0],
                    'Close': symbol_data[symbol_data['Date'] == date_before_expiration]['Close'].values[0]
                })
                break

    prices_df1 = pd.DataFrame(data_16_days_before)
    prices_df1['Date'] = pd.to_datetime(prices_df1['Date'])
    prices_df1['Date'] = prices_df1['Date'] + pd.Timedelta(days=16)

    prices_df1.set_index('Date', inplace=True)
    hourly_df3 = prices_df1.resample('H').ffill()
    hourly_df3.drop(columns=["Symbol", "Expiration Year", "Expiration Month"], inplace=True)
    hourly_df3.rename(columns={"Open": "Open_16days_ago", "High": "High_16days_ago", "Close": "Close_16days_ago"}, inplace=True)
   
    # Merge and finalize
    hourly_df.reset_index(inplace=True)
    hourly_mean_df.reset_index(inplace=True)
    hourly_df3.reset_index(inplace=True)
   
    hourly_df.rename(columns={"Date": "datetime"}, inplace=True)
    hourly_mean_df.rename(columns={"expiration_date": "datetime"}, inplace=True)
    hourly_df3.rename(columns={"Date": "datetime"}, inplace=True)
   
    hourly_df.set_index("datetime", inplace=True)
    hourly_df = pd.merge(hourly_df, hourly_mean_df, on='datetime', how='inner')
    hourly_df = pd.merge(hourly_df, hourly_df3, on="datetime")
    hourly_df.drop(columns=["Symbol", "Expiration Year", "Expiration Month"], inplace=True)
    gas_prices = hourly_df.copy()
    last_date = gas_prices['datetime'].max()
    new_dates = pd.date_range(start=last_date, periods=30*24+1, freq='H')[1:]  # exclude the last_date itself

    # Duplicate the last row and adjust the datetime values
    last_row = gas_prices.iloc[-1:].copy()
    new_rows = pd.concat([last_row] * len(new_dates), ignore_index=True)
    new_rows['datetime'] = new_dates

    # Concatenate the new rows with the original DataFrame
    gas_prices_extended = pd.concat([gas_prices, new_rows], ignore_index=True)
    return gas_prices_extended





# Prepare Coal Data
def treat_Coal_data(file_path):
    # Read data
    Coal_Prices = pd.read_csv(file_path)
    
    # Mapping of month codes to numbers
    month_codes = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4,
        'K': 5, 'M': 6, 'N': 7, 'Q': 8,
        'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }

    # Function to extract expiration date from symbol
    def extract_expiration_date(symbol):
        month_code = symbol[-3]  # Second to last character for month code
        year_str = '20' + symbol[-2:]  # Last two characters for year
        month = month_codes[month_code]
        expiration_date = pd.to_datetime(f'{year_str}-{month:02d}-01')  # Set to first day of the month
        return expiration_date

    # Apply expiration date extraction
    Coal_Prices['expiration_date'] = Coal_Prices['Symbol'].apply(extract_expiration_date)

    # Convert Date column to datetime
    Coal_Prices["Date"] = pd.to_datetime(Coal_Prices["Date"])

    # Calculate mean prices based on 4-month pre-expiration period
    mean_prices = []
    for exp_date in Coal_Prices['expiration_date'].unique():
        start_date = exp_date - pd.DateOffset(months=4)
        end_date = exp_date - pd.DateOffset(months=1)
        
        # Mask for filtering dates in the 4-month window
        mask = (Coal_Prices['Date'] >= start_date) & (Coal_Prices['Date'] <= end_date)
        mean_price_close = Coal_Prices.loc[mask, 'Close'].mean()
        mean_price_high = Coal_Prices.loc[mask, 'High'].mean()
        mean_price_open = Coal_Prices.loc[mask, 'Open'].mean()
        
        # Append results
        mean_prices.append({
            'expiration_date': exp_date,
            'mean_price_close_Coal': mean_price_close,
            'mean_price_open_Coal': mean_price_open,
            'mean_price_high_Coal': mean_price_high
        })

    # Convert to DataFrame and sort by expiration date
    mean_prices_df_Coal = pd.DataFrame(mean_prices).sort_values(by="expiration_date")

    # Resample to hourly frequency, filling forward
    mean_prices_df_Coal['expiration_date'] = pd.to_datetime(mean_prices_df_Coal['expiration_date'])
    mean_prices_df_Coal.set_index('expiration_date', inplace=True)
    hourly_mean_df_Coal = mean_prices_df_Coal.resample('H').ffill()

    # Reset index and rename columns for clarity
    hourly_mean_df_Coal.reset_index(inplace=True)
    hourly_mean_df_Coal.rename(columns={"expiration_date": "datetime"}, inplace=True)
    hourly_mean_df_Coal["datetime"] = pd.to_datetime(hourly_mean_df_Coal["datetime"])

    # Return the structured DataFrame
    return hourly_mean_df_Coal



import pandas as pd

def treat_EUA_data(file_path):
    # Read data and drop unwanted columns
    EUA = pd.read_csv(file_path)
    EUA.drop(columns=["id"], inplace=True)

    # Month codes for mapping expiration months
    month_codes = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4,
        'K': 5, 'M': 6, 'N': 7, 'Q': 8,
        'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }

    # Function to extract expiration date from symbol
    def extract_expiration_date(symbol):
        month_code = symbol[-3]  # Second to last character for month code
        year_str = '20' + symbol[-2:]  # Last two characters for year
        month = month_codes[month_code]
        expiration_date = pd.to_datetime(f'{year_str}-{month:02d}-01')  # Set to first day of the month
        return expiration_date

    # Apply expiration date extraction
    EUA['expiration_date'] = EUA['Symbol'].apply(extract_expiration_date)

    # Extract expiration month and year
    def extract_expiration_info(symbol):
        month_symbol = symbol[2]  # Month code
        year = '20' + symbol[3:]  # Year code
        month = month_codes.get(month_symbol)
        return month, int(year)

    EUA['expiration_month'], EUA['expiration_year'] = zip(*EUA['Symbol'].apply(extract_expiration_info))

    # Convert Date column to datetime
    EUA["Date"] = pd.to_datetime(EUA["Date"])

    # Calculate mean prices based on 4-month pre-expiration period
    mean_prices = []
    for exp_date in EUA['expiration_date'].unique():
        start_date = exp_date - pd.DateOffset(months=4)
        end_date = exp_date - pd.DateOffset(months=1)
        
        # Mask for filtering dates in the 4-month window
        mask = (EUA['Date'] >= start_date) & (EUA['Date'] <= end_date)
        mean_price_close = EUA.loc[mask, 'Close'].mean()
        mean_price_high = EUA.loc[mask, 'High'].mean()
        mean_price_open = EUA.loc[mask, 'Open'].mean()
        
        # Append results
        mean_prices.append({
            'expiration_date': exp_date,
            'mean_price_close_EUA': mean_price_close,
            'mean_price_open_EUA': mean_price_open,
            'mean_price_high_EUA': mean_price_high
        })

    # Convert to DataFrame and sort by expiration date
    mean_prices_df = pd.DataFrame(mean_prices).sort_values(by="expiration_date")

    # Resample to hourly frequency, filling forward
    mean_prices_df['expiration_date'] = pd.to_datetime(mean_prices_df['expiration_date'])
    mean_prices_df.set_index('expiration_date', inplace=True)
    hourly_mean_df_EUA = mean_prices_df.resample('H').ffill()

    # Reset index and rename columns for clarity
    hourly_mean_df_EUA.reset_index(inplace=True)
    hourly_mean_df_EUA.rename(columns={"expiration_date": "datetime"}, inplace=True)
    hourly_mean_df_EUA["datetime"] = pd.to_datetime(hourly_mean_df_EUA["datetime"])

    # Return the hourly resampled DataFrame
    return hourly_mean_df_EUA


# Prepare Gas data for model Training
def prepare_gas_and_cap_data_for_model_train(power_file,capacity_file=None):
    # Load and preprocess Power data
    Power = pd.read_csv(power_file)
    Power = Power.iloc[1:]
    Power.rename(columns={"Unnamed: 0": "datetime"}, inplace=True)
    Power = Power[["Fossil Brown coal/Lignite", "Fossil Gas", "Fossil Hard coal", "datetime"]]
    Power['datetime'] = pd.to_datetime(Power['datetime'], utc=True)
    Power.set_index('datetime', inplace=True)
    Power = Power.astype(float)
    print(Power)
    hourly_df = Power.resample('h').sum()

    hourly_df.reset_index(inplace=True)
    hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
    #hourly_df.set_index('datetime', inplace=True)
    Power.reset_index(inplace=True)
    

    # Define season determination function
   

    # Generate gas dataframe
    gas = hourly_df[["datetime", "Fossil Gas","Fossil Hard coal","Fossil Brown coal/Lignite"]].copy()
    gas['month'] = gas['datetime'].dt.month
   # gas['season'] = gas['month'].apply(season_determination)

    # Load Capacity data and set datetime
    Capacity = pd.read_csv(capacity_file)
    Capacity.rename(columns={"Unnamed: 0": "datetime"}, inplace=True)
    Capacity['datetime'] = pd.to_datetime(Capacity['datetime'])

    # Prepare data for polynomial regression and capacity approximation
    start_date = Capacity['datetime'].min()
    end_date = Capacity['datetime'].max()
    hourly_timestamps = pd.date_range(start=start_date, end=end_date + timedelta(days=1), freq='H')

    X = (Capacity['datetime'] - start_date).dt.total_seconds().values.reshape(-1, 1) / (3600 * 24 * 365.25)  # Convert to years
    y = Capacity['Fossil Gas'].values

    # Polynomial regression model
    degree = 5
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Generate predictions
    hourly_X = (hourly_timestamps - start_date).total_seconds().values.reshape(-1, 1) / (3600 * 24 * 365.25)
    hourly_X_poly = poly.transform(hourly_X)
    hourly_predictions = model.predict(hourly_X_poly)

    # Create Gas_capacity_approximated dataframe
    Gas_capacity_approximated = pd.DataFrame({'datetime': hourly_timestamps, 'Fossil Gas': hourly_predictions})

    return gas, Gas_capacity_approximated


# Create Residuals For Training:

def create_residuals(load,generation):
    data=pd.merge(load,generation,on="datetime",how="left")
    data["sum"]=data["Solar"]+data["Wind Offshore"]+data["Wind Onshore"]+data["Hydro Run-of-river and poundage"]
    data["residuals"]=data["load"]-data["sum"]
    data=data[["residuals","Fossil Gas","Fossil Brown coal/Lignite","Fossil Hard coal"]]
    return data

def season_determination(month):
        if month in [6, 7, 8, 9]:
            return "summer"
        elif month in [1, 2, 12]:
            return "winter"
        else:
            return "spring/fall"
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
def create_data_for_model(Residuals,hourly_mean_df_EUA,hourly_mean_df_Coal,gas_prices):
    data=pd.merge(Residuals,hourly_mean_df_Coal,on="datetime")
    data=pd.merge(data,hourly_mean_df_EUA,on="datetime")
    data=pd.merge(data,gas_prices,on="datetime")
    data=Pre_Processing(data)
    data['Gas_Cost'] = 0.5 * data['mean_price_open_EUA'] + data["mean_price_open"]

# Creating the Coal_Cost feature
    data['Coal_Cost'] = 1.1 * data['mean_price_open_EUA'] + data['mean_price_open_Coal']

# Creating the Difference feature (Coal_Cost - Gas_Cost)
    data['Cost_Difference'] = data['Coal_Cost'] - data['Gas_Cost']

# Creating the Boolean feature (Coal_Cost > Gas_Cost)
    data['Is_Coal_Cost_Higher'] = data['Coal_Cost'] > data['Gas_Cost']

    return data
