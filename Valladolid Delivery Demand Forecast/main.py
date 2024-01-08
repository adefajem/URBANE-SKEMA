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
    bike_forecast, car_forecast = functions.time_series_analysis(historical_data, num_days_forecast)
    
    # Clustering
    addresses_data = pd.read_excel('input/deliveries_data_as_is_UC3_initial.xlsx')
    # Filter out NAs just in case addresses_data = addresses_data.dropna()
    # Remove rows where the Longitude and/or Latitude are null values
    addresses_data.dropna(axis=0, how='any',subset=['Delivery latitude','Delivery longtitude'],inplace=True)
    # Cluster and get centroids
    addresses_data, centroids = functions.cluster_addresses(num_clusters, addresses_data)
    
    # Get the probabilities of deliveries going to a cluster using either bike or car
    bike_probabilities, car_probabilities = functions.get_location_probabilities(addresses_data, num_clusters)
    
    
    
    
    # --- Output ---
    bike_forecast_final, car_forecast_final = functions.output_complete_forecast(addresses_data, bike_forecast, car_forecast, centroids, bike_probabilities, car_probabilities)
    # Write  to file
    car_forecast_final.to_excel('output/deliveries_data_as_is_UC3_car_forecast.xlsx', index=False)
    bike_forecast_final.to_excel('output/deliveries_data_as_is_UC3_bike_forecast.xlsx', index=False)
    
    
    
if __name__ == "__main__":
    main()
