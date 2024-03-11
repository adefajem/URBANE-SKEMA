# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:16:11 2024

@author: ade.fajemisin
"""

import pandas as pd
import functions

def main():
    
    # --- Inputs ---
    # Historical Data
    historical_data = pd.read_excel('input/deliveries_data_as_is_UC3_initial.xlsx')
    input_params = pd.read_excel('input/input_params.xlsx')
    num_days_forecast = int(input_params['Number of days to forecast'])
    num_clusters = int(input_params['Number of clusters to use'])
       
    # --- Forecasting ---
    # Time series analysis
    all_vehicle_forecasts = functions.time_series_analysis(historical_data, num_days_forecast)
    
    # Clustering
    addresses_data = pd.read_excel('input/deliveries_data_as_is_UC3_initial.xlsx')
    # Filter out NAs just in case
    # Remove rows where the Longitude and/or Latitude are null values
    addresses_data.dropna(axis=0, how='any',subset=['Delivery latitude','Delivery longtitude'],inplace=True)    
    # Cluster and get centroids
    addresses_data, centroids = functions.cluster_addresses(num_clusters, addresses_data)
    
    # Get the probabilities of deliveries going to a cluster using either bike or car
    all_vehicle_probabilities = functions.get_location_probabilities(addresses_data, num_clusters)
    
    
    # --- Output ---
    all_forecasts_final = functions.output_complete_forecast(addresses_data, all_vehicle_forecasts, all_vehicle_probabilities, centroids)
    
    # Write to file
    vehicle_types = list(addresses_data['Vehicle type'].unique())    
    for f in range(len(all_forecasts_final)):
        output_filename = 'output/deliveries_data_as_is_UC3_'+ vehicle_types[f] +'_forecast.xlsx'
        all_forecasts_final[f].to_excel(output_filename, index=False)
    
    
    
if __name__ == "__main__":
    main()
