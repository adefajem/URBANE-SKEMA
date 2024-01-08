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
    
    # Filter out NAs just in case
    historical_data = historical_data.dropna()
    # Drop Delivery_ID column
    historical_data.drop(['Delivery_ID'], axis=1, inplace=True)
    
    # Split into car and bike sets
    
    # ---------------------------- BIKE FORECAST ----------------------------------
    bike_data_ts = historical_data.loc[historical_data["Vehicle type"] == 'Bike', :]
    bike_data_ts = bike_data_ts.sort_values('Date of delivery')
    bike_data_ts = bike_data_ts.groupby('Date of delivery')['Total parcels/letters'].sum().reset_index()

    # Rename
    bike_data_ts = bike_data_ts.rename(columns={'Date of delivery': 'ds', 'Total parcels/letters': 'y'})

    # Hyperparameter tuning
    # There is no hyperparameter tuning done yet. The amount of data we have currently is too litte
    # for this. We can add this functionality in future when more data becomes available.
    
    bike_ts_model = Prophet(interval_width=0.95)
    bike_ts_model.fit(bike_data_ts)

    # Forecast
    bike_ts_forecast = bike_ts_model.make_future_dataframe(periods=num_days_forecast, freq='D')
    bike_ts_forecast = bike_ts_model.predict(bike_ts_forecast)

    bike_forecast = bike_ts_forecast[['ds','yhat']].iloc[-num_days_forecast:]
    bike_forecast = bike_forecast.astype({'yhat':'int'})
    bike_forecast = bike_forecast.reset_index(drop=True)
    
    # ---------------------------- CAR FORECAST ----------------------------------

    car_data_ts = historical_data.loc[historical_data["Vehicle type"] == 'Car', :]
    car_data_ts = car_data_ts.sort_values('Date of delivery')
    car_data_ts = car_data_ts.groupby('Date of delivery')['Total parcels/letters'].sum().reset_index()
    
    # Rename
    car_data_ts = car_data_ts.rename(columns={'Date of delivery': 'ds', 'Total parcels/letters': 'y'})

    # Hyperparameter tuning
    # There is no hyperparameter tuning done yet. The amount of data we have currently is too litte
    # for this. We can add this functionality in future when more data becomes available.
    
    car_ts_model = Prophet(interval_width=0.95)
    car_ts_model.fit(car_data_ts)

    # Forecast
    car_ts_forecast = car_ts_model.make_future_dataframe(periods=num_days_forecast, freq='D')
    car_ts_forecast = car_ts_model.predict(car_ts_forecast)
    
    car_forecast = car_ts_forecast[['ds','yhat']].iloc[-num_days_forecast:]
    car_forecast = car_forecast.astype({'yhat':'int'})
    car_forecast = car_forecast.reset_index(drop=True)

    # Harmonize dates
    bike_forecast, car_forecast = harmonize_dates(bike_forecast, car_forecast)
    
    # Return forecast
    return bike_forecast, car_forecast

def harmonize_dates(bike_forecast, car_forecast):
    # Convert date columns to datetime objects
    bike_forecast['ds'] = pd.to_datetime(bike_forecast['ds'])
    car_forecast['ds'] = pd.to_datetime(car_forecast['ds'])
    
    # Identify the latest date in each dataframe
    latest_date_bike = bike_forecast['ds'].max()
    latest_date_car = car_forecast['ds'].max()
    
    # Determine which dataframe has the latest dates
    if latest_date_bike > latest_date_car:
        latest_df = bike_forecast
        other_df = car_forecast
    else:
        latest_df = car_forecast
        other_df = bike_forecast
    
    # Replace the dates in the other dataframe with the ones from the dataframe with the latest dates
    other_df['ds'] = latest_df['ds']
    
    return bike_forecast, car_forecast

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
    bike_addresses = addresses_data.loc[addresses_data["Vehicle type"] == 'Bike', :]
    bike_addresses = bike_addresses.reset_index()
    bike_nums = bike_addresses['Cluster label'].value_counts()   
    bike_probabilities = calc_probabilities(num_clusters, bike_nums)
    bike_probabilities = list(bike_probabilities.loc[0])
    
    car_addresses = addresses_data.loc[addresses_data["Vehicle type"] == 'Car', :]
    car_addresses = car_addresses.reset_index()
    car_nums = car_addresses['Cluster label'].value_counts()
    car_probabilities = calc_probabilities(num_clusters, car_nums)   
    car_probabilities = list(car_probabilities.loc[0])
    
    return bike_probabilities, car_probabilities

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

def output_complete_forecast(addresses_data, bike_forecast, car_forecast, centroids, bike_probabilities, car_probabilities):    
    bike_forecast_rows = get_forecast_df(addresses_data, bike_forecast, bike_probabilities, centroids, 'Bike')
    car_forecast_rows = get_forecast_df(addresses_data, car_forecast, car_probabilities, centroids, 'Car')
    
    # Bike
    bike_forecast_final = pd.DataFrame(data=bike_forecast_rows, columns=['Delivery_ID','Date of delivery','Delivery time slot','Delivery latitude','Delivery longtitude','Method of Delivery', 
'Total parcels/letters','Vehicle type'])
    bike_forecast_final.sort_values(by='Date of delivery', inplace = True) 
    # Filter out rows where the values in 'Total parcels/letters' are zero
    bike_forecast_final = bike_forecast_final[bike_forecast_final['Total parcels/letters'] != 0]    
    bike_forecast_final = bike_forecast_final.reset_index(drop=True)    
    bike_forecast_final['Delivery_ID'] = list(range(len(bike_forecast_final)))
    
     # Car
    car_forecast_final = pd.DataFrame(data=car_forecast_rows, columns=['Delivery_ID','Date of delivery','Delivery time slot','Delivery latitude','Delivery longtitude','Method of Delivery', 
'Total parcels/letters','Vehicle type'])
    car_forecast_final.sort_values(by='Date of delivery', inplace = True) 
    # Filter out rows where the values in 'Total parcels/letters' are zero
    car_forecast_final = car_forecast_final[car_forecast_final['Total parcels/letters'] != 0]    
    car_forecast_final = car_forecast_final.reset_index(drop=True)    
    car_forecast_final['Delivery_ID'] = list(range(len(car_forecast_final)))
       
    return bike_forecast_final, car_forecast_final