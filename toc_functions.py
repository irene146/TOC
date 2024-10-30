import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from glob import glob
import os


def merge_picarro_files(root_folder):
    #list to store data
    all_data = []

    # loop through day subfolders
    for day_folder in os.listdir(root_folder):
        day_path = os.path.join(root_folder, day_folder)

        #loop through files of each day folder
        for file_name in os.listdir(day_path):
            file_path = os.path.join(day_path, file_name)
        
        # read .dat using fwf
            df = pd.read_fwf(file_path)
            
            # append data 
            all_data.append(df)

    # combine into a df
    merged_df = pd.concat(all_data, ignore_index=True)

    # create datetime column 
    merged_df['datetime'] = pd.to_datetime(merged_df['DATE'] + ' ' + merged_df['TIME'], errors='coerce')

    # drop date and time columns 
    merged_df.drop(columns=['DATE', 'TIME'], inplace=True)

    #  datetime first
    columns = ['datetime'] + [col for col in merged_df.columns if col != 'datetime']
    merged_df = merged_df[columns]

    return merged_df


def qc(df):
    
    df['CO2_corrected'] = (df['CO2_dry'] + 0.63141) / 0.99357
    df['CH4_corrected'] = df['CH4_dry'] * (2.024799 / 2.0238)
    return df
    



def TOC_df(df):
    """
    Makes df for TOC calculations, averages catalyst and ambient times
    
    Args:
        df: df with qc measurements
    
    Returns:
        df to do TOC calculations 
    """
    # lists for storing averages
    avg_times = []
    avg_co2_ambient = []
    avg_co2_catalyst = []
    avg_ch4_ambient = []
    avg_ch4_catalyst = []
    avg_co_ambient = []
    avg_co_catalyst = []

    # detect for valve changes
    valve_change = df[(df['solenoid_valves'] != 2.0) & (df['solenoid_valves'] != 0.0)]  # valve =2 is catalyst
                                                                                       # valve = 0 is ambient

    # averages before valve changes
    for i in valve_change.index:
        #  time window (25s before valve change, ending 2s before)
        end_time = i - pd.Timedelta(seconds=2)
        start_time = end_time - pd.Timedelta(seconds=25)
        
        # get data in the main df within time window
        window_df = df[(df.index > start_time) & (df.index < end_time)]
        
        # averages
        avg_co2 = window_df['CO2_corrected'].mean()
        avg_ch4 = window_df['CH4_corrected'].mean()
        avg_co = window_df['CO'].mean()
        avg_valve = window_df['solenoid_valves'].mean()
        avg_time_val = window_df.index.mean()
        avg_times.append(avg_time_val)

        # sort appropriate lists based on valve state
        if avg_valve == 2.0:  # catalyst
            avg_co2_catalyst.append(avg_co2)
            avg_ch4_catalyst.append(avg_ch4)
            avg_co_catalyst.append(avg_co)
            avg_co2_ambient.append(np.nan)
            avg_ch4_ambient.append(np.nan)
            avg_co_ambient.append(np.nan)
        elif avg_valve == 0.0:  # ambient
            avg_co2_ambient.append(avg_co2)
            avg_ch4_ambient.append(avg_ch4)
            avg_co_ambient.append(avg_co)
            avg_co2_catalyst.append(np.nan)
            avg_ch4_catalyst.append(np.nan)
            avg_co_catalyst.append(np.nan)

    # create results df
    TOC_df = pd.DataFrame({
        'datetime': avg_times,
        'avg_co2_ambient': avg_co2_ambient,
        'avg_ch4_ambient': avg_ch4_ambient,
        'avg_co_ambient': avg_co_ambient,
        'avg_co2_catalyst': avg_co2_catalyst,
        'avg_ch4_catalyst': avg_ch4_catalyst,
        'avg_co_catalyst': avg_co_catalyst
    })

    return TOC_df.set_index('datetime')

def calculate_toc(TOC_df):
    """
    calculate final TOC values 
    
    Args:
        TOC_df: DataFrame with averaged values
    
    Returns:
        DataFrame with TOC results
    """
    toc_results = []

    # process in pairs (ambient + catalyst)
    for i in range(0, len(TOC_df), 2):
        pair = TOC_df.iloc[i:i+2]
        # sum catalyst and ambient values
        sum_catalyst = pair[['avg_co2_catalyst', 'avg_ch4_catalyst', 'avg_co_catalyst']].sum().sum()
        sum_ambient = pair[['avg_co2_ambient', 'avg_ch4_ambient', 'avg_co_ambient']].sum().sum()
        # calculate TOC
        toc = sum_catalyst - sum_ambient
        # store result with mean timestamp
        toc_results.append((pair.index.mean(), toc))

    return pd.DataFrame(toc_results, columns=['datetime', 'TOC']).set_index('datetime')