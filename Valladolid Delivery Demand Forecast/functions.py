# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:18:10 2024

@author: ade.fajemisin
"""

import pandas as pd
from prophet import Prophet
#from prophet.diagnostics import cross_validation
#from prophet.diagnostics import performance_metrics
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def time_series_analysis(historical_data, num_days_forecast):
    all_vehicle_forecasts = []
    
    # Filter out NAs just in case
    historical_data = historical_data.dropna()
    # Drop Delivery_ID column
    historical_data.drop(['Delivery_ID'], axis=1, inplace=True)
    
    # Get a list of vehicle types
    vehicle_types = list(historical_data['Vehicle type'].unique())
    
    for vehicle in vehicle_types:
        # Subset data, sort and group
        vehicle_data_ts = historical_data.loc[historical_data["Vehicle type"] == vehicle, :]
        vehicle_data_ts = vehicle_data_ts.sort_values('Date of delivery')
        vehicle_data_ts = vehicle_data_ts.groupby('Date of delivery')['Total parcels/letters'].sum().reset_index()
        
        # Rename columns
        vehicle_data_ts = vehicle_data_ts.rename(columns={'Date of delivery': 'ds', 'Total parcels/letters': 'y'})
        
        # Hyperparameter tuning
        # There is no hyperparameter tuning done yet. The amount of data we have currently is too litte
        # for this. We can add this functionality in future when more data becomes available.
        
        # Model fitting
        vehicle_ts_model = Prophet(interval_width=0.95)
        vehicle_ts_model.fit(vehicle_data_ts)
        
        # Forecast
        vehicle_ts_forecast = vehicle_ts_model.make_future_dataframe(periods=num_days_forecast, freq='D')
        vehicle_ts_forecast = vehicle_ts_model.predict(vehicle_ts_forecast)

        vehicle_forecast = vehicle_ts_forecast[['ds','yhat']].iloc[-num_days_forecast:]
        vehicle_forecast = vehicle_forecast.astype({'yhat':'int'})
        vehicle_forecast = vehicle_forecast.reset_index(drop=True)
        
       
        # Save
        all_vehicle_forecasts.append(vehicle_forecast)  
    
    # Harmonize dates
    all_vehicle_forecasts = harmonize_dates(all_vehicle_forecasts)
       
    return all_vehicle_forecasts

def harmonize_dates(all_vehicle_forecasts):
    latest_dates_all = []    
    for vehicle_forecast in all_vehicle_forecasts:
        # Convert date columns to datetime objects
        vehicle_forecast['ds'] = pd.to_datetime(vehicle_forecast['ds'])
        
        # Identify the latest date in each dataframe
        latest_date_vehicle = vehicle_forecast['ds'].max()        
        latest_dates_all.append(latest_date_vehicle)
    
    # Determine which dataframe has the latest dates
    latest_index = latest_dates_all.index(max(latest_dates_all))
    latest_df = all_vehicle_forecasts[latest_index]
    
    # Replace the dates in the other dataframe with the ones from the dataframe with the latest dates
    for vehicle_forecast in all_vehicle_forecasts:
        vehicle_forecast['ds'] = latest_df['ds']
        
    return all_vehicle_forecasts

def cluster_addresses(num_clusters, addresses_data):    
    kmeans = KMeans(n_clusters = 3, init ='k-means++')
    kmeans.fit(addresses_data[addresses_data.columns[3:5]]) # Compute k-means clustering.
    addresses_data['Cluster label'] = kmeans.fit_predict(addresses_data[addresses_data.columns[3:5]])
    centroids = kmeans.cluster_centers_ # Coordinates of cluster centers.
    centroids_tuples = [tuple(row) for row in centroids]
    
    return addresses_data, centroids_tuples

def calc_probabilities(num_clusters, vehicle_numbers):    
    headers = []
    probabilities = []
    for i in range(num_clusters):
        col_names = 'Cluster_' + str(i) + '_pct'
        headers.append(col_names)    
        probabilities.append([vehicle_numbers[i] / sum(vehicle_numbers)])

    ddd = dict(zip(headers, probabilities))
    location_probabilities = pd.DataFrame(data=ddd)
    
    return location_probabilities

def get_location_probabilities(addresses_data, num_clusters):
    all_vehicle_probabilities = []
    
    # Get a list of vehicle types
    vehicle_types = list(addresses_data['Vehicle type'].unique())
    
    # Get location probabilities for each vehicle type
    for vehicle in vehicle_types:
        vehicle_addresses = addresses_data.loc[addresses_data['Vehicle type'] == vehicle, :]
        vehicle_addresses = vehicle_addresses.reset_index()
        vehicle_nums = vehicle_addresses['Cluster label'].value_counts()   
        vehicle_probabilities = calc_probabilities(num_clusters, vehicle_nums)
        vehicle_probabilities = list(vehicle_probabilities.loc[0])
        # Store
        all_vehicle_probabilities.append(vehicle_probabilities)
    
    return all_vehicle_probabilities

def divide_with_probabilities(total, probabilities):
    parts = [int(prob * total) for prob in probabilities]

    # Adjust for rounding errors
    remaining = total - sum(parts)
    parts[0] += remaining

    return parts

def get_forecast_df(addresses_data, forecast, probabilities,centroids, vehicle_type):
    all_data = []

    for i in range(len(forecast)):
        # Divide into parts
        forecast_areas = divide_with_probabilities(forecast.loc[i]['yhat'], probabilities)

        # Add rows into dataset
        for j in range(len(forecast_areas)):
            temp = []
            temp.append('')
            temp.append(forecast.loc[i]['ds'].to_pydatetime().strftime('%Y-%m-%d'))
            temp.append(addresses_data.loc[0]['Delivery time slot'])
            temp.append(centroids[j][0])
            temp.append(centroids[j][1])
            temp.append('Home')
            temp.append(forecast_areas[j])
            temp.append(vehicle_type)

            all_data.append(temp)
            
    return all_data

def output_complete_forecast(addresses_data, all_vehicle_forecasts, all_vehicle_probabilities, centroids):    
    all_forecasts_final = []
    
    vehicle_types = list(addresses_data['Vehicle type'].unique())
    
    for v in range(len(all_vehicle_forecasts)):
        vehicle_forecast_rows = get_forecast_df(addresses_data, all_vehicle_forecasts[v], all_vehicle_probabilities[v], centroids, vehicle_types[v])
        vehicle_forecast_final = pd.DataFrame(data=vehicle_forecast_rows, columns=['Delivery_ID','Date of delivery','Delivery time slot','Delivery latitude','Delivery longtitude','Method of Delivery', 
'Total parcels/letters','Vehicle type'])
        vehicle_forecast_final.sort_values(by='Date of delivery', inplace = True) 
         # Filter out rows where the values in 'Total parcels/letters' are zero
        vehicle_forecast_final = vehicle_forecast_final[vehicle_forecast_final['Total parcels/letters'] != 0]    
        vehicle_forecast_final = vehicle_forecast_final.reset_index(drop=True)    
        vehicle_forecast_final['Delivery_ID'] = list(range(len(vehicle_forecast_final)))
        # Store
        all_forecasts_final.append(vehicle_forecast_final)
       
    return all_forecasts_final

