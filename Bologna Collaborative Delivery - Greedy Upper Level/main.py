# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:26:53 2024

@author: ade.fajemisin
"""

import pandas as pd
import numpy as np
import functions
import optimization


def main():
    # --- Inputs ---
    city_instance_df = pd.read_excel('input/problem_instance.xlsx')
    electricity_inputs = pd.read_excel('input/electricity_generation_breakdown.xlsx')
    emission_factor = list(electricity_inputs['Emission Factor'])
    generation_percentage = list(electricity_inputs['Generation Percentage'])

    # For an electric delivery vehicle k of the type:
    # https://scoobic.com/en/especificaciones/scoobic-light-en/
    # Battery capacity of electric vehicle in kWh
    bc = pd.read_excel('input/battery_capacity.xlsx')
    battery_capacity = int(bc['Capacity (kWh)'])


    # Number of unique nodes
    num_nodes = len(city_instance_df)

    # Get total number of depots and lockers
    num_depots = len(city_instance_df.loc[city_instance_df['type'].str.contains('depot')])
    num_lockers = len(city_instance_df.loc[city_instance_df['type']=='locker'])
    total_num_depots_lockers = num_depots + num_lockers

    # Get number of packages
    num_packages = len(city_instance_df.loc[city_instance_df['type']=='package'])
    packages = list(range(num_packages))

    # Destinations of parcels is in the order give in city_instance_df: parcel 0 -> node 5, parcel 1 -> node 6, etc.
    destinations = [i+total_num_depots_lockers for i in packages]

    # Satellite nodes
    locker_nodes = list(city_instance_df.loc[city_instance_df['type']=='locker']['node'])

    # Locker capacities
    oc = pd.read_excel('input/locker_capacities.xlsx')
    original_locker_capacities = list(oc['Capacity'])
    locker_capacities = {}
    for i in range(len(locker_nodes)):
        locker_capacities[locker_nodes[i]] = original_locker_capacities[i]
        

    # DSP depots, nodes and arcs
    depots_df = city_instance_df.loc[city_instance_df['type'].str.contains('depot')]
    dsp_depots = []
    dsp_d_nodes = []
    # Assuming only one depot per DSP
    for d in range(len(depots_df)):
        dsp_depots.append(int(depots_df['node'][d]))
        dsp_d_nodes.append([int(depots_df['node'][d])] + locker_nodes + destinations)
        
    dsp_d_arcs = []
    for d in range(len(dsp_d_nodes)):
        arcs = [(i, j) for i in dsp_d_nodes[d] for j in dsp_d_nodes[d] if i!=j]
        dsp_d_arcs.append(arcs)
        
    # DSP vehicles
    lmd = pd.read_excel('input/last_milers.xlsx')
    num_vehicles_per_DSP = list(lmd['num_vehicles'])
    num_DSPs = len(lmd['num_vehicles'])
    # we assume that each DSP has enough capacity. 
    each_dsp_cap = np.ceil((num_packages/num_DSPs))# + (num_packages/10))
    dsp_capacities = []
    for i in range(num_DSPs):
        dsp_capacities.append(each_dsp_cap)

    # Get locker preferences
    # rankings = get_locker_rankings(city_instance_df, locker_nodes, delivery_nodes)
    rankings = functions.get_locker_rankings(city_instance_df, locker_nodes, destinations)

    # Assign to lockers
    locker_assignments = functions.assign_packages_to_lockers(packages, rankings, original_locker_capacities, locker_nodes)
    lamda = functions.generate_lamda(locker_assignments, num_packages, num_nodes)


    start_time_periods = [7]
    end_time_periods = [23]
    time_periods_multiplier = [1] # Only 1 time period for now...
    time_violation_penalty = 1000

    # Problem runtime time limit is also read from file
    tim = pd.read_excel('input/time_limit.xlsx')
    time_limit_seconds = int(tim['Time Limit Per Follower (seconds)'])



    # --- Run Optimizer ---
    distance_matrix, travel_time_matrix = functions.generate_distance_time_mat(city_instance_df, time_periods_multiplier)
    leave_time_start, leave_time_end, arrive_time, earliest, latest, bigM_matrix = functions.generate_all_times_and_bigMs(city_instance_df, start_time_periods,end_time_periods, travel_time_matrix)

    num_time_periods_matrix = functions.get_num_time_periods(distance_matrix)

    emissions_matrix_EV = functions.compute_emissions_EV(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, num_time_periods_matrix, num_nodes)

    y_dsp_rankings = functions.get_DSP_rankings(city_instance_df, dsp_depots, destinations)
    y = functions.assign_packages_to_DSPs(packages, y_dsp_rankings, dsp_capacities)



    routing_solutions = optimization.solve_param_last_mile_followers(y, lamda, dsp_d_nodes, dsp_d_arcs,
                                    dsp_depots, locker_nodes, packages, destinations, 
                                    num_vehicles_per_DSP, distance_matrix, travel_time_matrix, 
                                    num_time_periods_matrix, bigM_matrix, earliest, latest, 
                                    leave_time_start, leave_time_end, emissions_matrix_EV, time_violation_penalty, 
                                    time_limit_seconds)

    # --- Write solution to file ---  
    res_filename = 'output/results.xlsx'
    functions.write_instance_results_to_file(res_filename, y, locker_assignments, routing_solutions[1], num_vehicles_per_DSP,
                                          distance_matrix, emissions_matrix_EV, dsp_depots, locker_nodes, packages, destinations)

    print('Done!')

if __name__ == "__main__":
    main()
