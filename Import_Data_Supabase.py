import pandas as pd
from supabase import Client, create_client
import os

# Function For Updating Prices Data
def fetch_data_by_date(supabase_url, supabase_key):
    # Create a Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Define table names
    tables = ["Gas_TTF", "EUA", "Coal"]
    data_frames = {}

    for table_name in tables:
        # Set the start date as the latest date in the existing file, if available
        file_path = f"data/{table_name}_rows.csv"
        if os.path.exists(file_path):
            old_df = pd.read_csv(file_path)
            old_df["Date"]=pd.to_datetime(old_df["Date"])
            start_date = old_df['Date'].max()+pd.Timedelta(days= 1)
        else:
            old_df = pd.DataFrame()
            start_date = "1970-01-01"  # Default earliest date if no file exists

        # Fetch data with the new starting date and limit of 1000 rows
        response = (
            supabase.table(table_name)
            .select("*")
            .gte("Date", start_date)  # Greater than or equal to the desired Start Date
            .order("Date", desc=True)
            .limit(1000)
            .execute()
        )
        
        # Convert the fetched data to a DataFrame
        new_df = pd.DataFrame(response.data)
        if(len(new_df)):
            new_df["Date"]=pd.to_datetime(new_df["Date"])
        
        # Merge the new data with the old data (if exists) and remove duplicates based on 'Date'
        if not old_df.empty:
            merged_df = pd.concat([old_df, new_df]).sort_values("Date")
        else:
            merged_df = new_df

        # Save the merged data to the CSV file
        merged_df.to_csv(file_path, index=False)
        
        # Add the merged DataFrame to the dictionary for returning
        data_frames[table_name] = merged_df
        


    return data_frames


