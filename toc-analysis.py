import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from glob import glob
import os


def merge_dat_files(root_folder):
    # List to store data from all files
    all_data = []

    # Loop through all subfolders named as day numbers
    for day_folder in os.listdir(root_folder):
        day_path = os.path.join(root_folder, day_folder)
        if os.path.isdir(day_path):
            # Find all DAT files in the current day folder
            dat_files = glob(os.path.join(day_path, '*.dat'))
            for file in dat_files:
                # Read the DAT file using fixed-width formatting (assuming fixed column widths)
                df = pd.read_fwf(file)
                # Add the DataFrame to the list
                all_data.append(df)

    # Combine all data into a single DataFrame
    merged_df = pd.concat(all_data, ignore_index=True)

    # Combine DATE and TIME into a single datetime column
    merged_df['datetime'] = pd.to_datetime(merged_df['DATE'] + ' ' + merged_df['TIME'], errors='coerce')

    # Drop original DATE and TIME columns
    merged_df.drop(columns=['DATE', 'TIME'], inplace=True)

    # Reorder columns to place datetime first
    columns = ['datetime'] + [col for col in merged_df.columns if col != 'datetime']
    merged_df = merged_df[columns]

    return merged_df

# Example usage
if __name__ == "__main__":
    root_folder = os.path.expanduser('~/data/')  # Update this path
    merged_df = merge_dat_files(root_folder)

    # Save the merged DataFrame to a CSV or other format if desired
    merged_df.to_csv('merged_data.csv', index=False)

def apply_corrections(df):
    """
    Apply specific correction factors to CO2 and CH4 measurements
    
    Args:
        df: DataFrame with raw measurements
    
    Returns:
        DataFrame with corrected values
    """
    # Apply exact correction factors from original code
    df['CO2_corrected'] = (df['CO2_dry'] + 0.63141) / 0.99357
    df['CH4_corrected'] = df['CH4_dry'] * (2.024799 / 2.0238)
    return df

def plot_diagnostic(df):
    """
    Create diagnostic plot of CO and cavity pressure
    
    Args:
        df: DataFrame with measurements
    """
    plt.figure()
    plt.plot(df.index, df.CO, label='co')
    plt.plot(df.index, df.CavityPressure, label='cavity pres')
    plt.legend()
    plt.show()

def calculate_toc_averages(df):
    """
    Calculate TOC averages based on valve positions
    
    Args:
        df: DataFrame with corrected measurements
    
    Returns:
        DataFrame with averaged values
    """
    # Initialize lists for storing averages
    avg_times = []
    avg_co2_ambient = []
    avg_co2_catalyst = []
    avg_ch4_ambient = []
    avg_ch4_catalyst = []
    avg_co_ambient = []
    avg_co_catalyst = []

    # Filter for valve changes
    valve_change = df[(df['solenoid_valves'] != 2.0) & (df['solenoid_valves'] != 0.0)]  # valve =2 is catalyst
                                                                                       # valve = 0 is ambient

    # Calculate averages before valve changes
    for i in valve_change.index:
        # Select time window (25s before valve change, ending 2s before)
        end_time = i - pd.Timedelta(seconds=2)
        start_time = end_time - pd.Timedelta(seconds=25)
        
        # Get data within time window
        window_df = df[(df.index > start_time) & (df.index < end_time)]
        
        # Calculate averages
        avg_co2 = window_df['CO2_corrected'].mean()
        avg_ch4 = window_df['CH4_corrected'].mean()
        avg_co = window_df['CO'].mean()
        avg_valve = window_df['solenoid_valves'].mean()
        avg_time_val = window_df.index.mean()
        avg_times.append(avg_time_val)

        # Sort into appropriate lists based on valve state
        if avg_valve == 2.0:  # Catalyst
            avg_co2_catalyst.append(avg_co2)
            avg_ch4_catalyst.append(avg_ch4)
            avg_co_catalyst.append(avg_co)
            avg_co2_ambient.append(np.nan)
            avg_ch4_ambient.append(np.nan)
            avg_co_ambient.append(np.nan)
        elif avg_valve == 0.0:  # Ambient
            avg_co2_ambient.append(avg_co2)
            avg_ch4_ambient.append(avg_ch4)
            avg_co_ambient.append(avg_co)
            avg_co2_catalyst.append(np.nan)
            avg_ch4_catalyst.append(np.nan)
            avg_co_catalyst.append(np.nan)

    # Create results DataFrame
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
    Calculate final TOC values from averaged data
    
    Args:
        TOC_df: DataFrame with averaged values
    
    Returns:
        DataFrame with TOC results
    """
    toc_results = []

    # Process in pairs (ambient + catalyst)
    for i in range(0, len(TOC_df), 2):
        pair = TOC_df.iloc[i:i+2]
        # Sum catalyst and ambient values
        sum_catalyst = pair[['avg_co2_catalyst', 'avg_ch4_catalyst', 'avg_co_catalyst']].sum().sum()
        sum_ambient = pair[['avg_co2_ambient', 'avg_ch4_ambient', 'avg_co_ambient']].sum().sum()
        # Calculate TOC
        toc = sum_catalyst - sum_ambient
        # Store result with timestamp
        toc_results.append((pair.index.mean(), toc))

    return pd.DataFrame(toc_results, columns=['datetime', 'TOC']).set_index('datetime')

# Example usage
if __name__ == "__main__":
    # Read data
    df = pd.read_csv('TOC_exp.csv', parse_dates=['datetime'], index_col='datetime')
    
    # Set analysis time range
    start_time = datetime.datetime(2024, 10, 24, 16, 30, 0)
    end_time = datetime.datetime(2024, 10, 25, 8, 15, 0)
    
    # Filter time range
    df = df.loc[start_time:end_time]
    
    # Apply corrections
    df = apply_corrections(df)
    
    # Create diagnostic plot
    plot_diagnostic(df)
    
    # Calculate TOC averages
    TOC_df = calculate_toc_averages(df)
    
    # Calculate final TOC
    toc = calculate_toc(TOC_df)
    
    # Save results
    toc.to_csv('toc_results.csv')
