# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:43:26 2024

@author: ade.fajemisin
"""
import pandas as pd
import numpy as np
import pickle
import functions
import optimization
from time import perf_counter as pc

def main():
    # --- Inputs ---
    with open('input/distance_times_27.pickle', 'rb') as handle:
        distance_and_times = pickle.load(handle)

    with open('input/locations_windows_27.pickle', 'rb') as handle:
        locations_and_windows = pickle.load(handle)
    
    city_instance_df = pd.read_excel('input/problem_instance_bilevel.xlsx')
    electricity_inputs = pd.read_excel('input/electricity_generation_breakdown.xlsx')
    emission_factor = list(electricity_inputs['Emission Factor'])
    generation_percentage = list(electricity_inputs['Generation Percentage'])
    
    
    # Get total number of depots and lockers
    num_depots = len(city_instance_df.loc[city_instance_df['type'].str.contains('depot')])
    num_lockers = len(city_instance_df.loc[city_instance_df['type']=='locker'])
    total_num_depots_lockers = num_depots + num_lockers
    
    # Get number of packages
    num_packages = len(city_instance_df.loc[city_instance_df['type']=='package'])
    packages = list(range(num_packages))
    
    # Get which packages come from which first-miler
    filtered_city = city_instance_df.dropna(subset=['package_id'])
    orig_dict = filtered_city[['package_id', 'first_miler_of_origin']].to_dict(orient='list')
    fm_origins = functions.get_firstmiler_package_origins(orig_dict)
    all_origins_bounds = []
    for i in range(len(fm_origins)):
        all_origins_bounds.append(len(fm_origins[i]))
    Pf = functions.create_bounds(all_origins_bounds)
    
    # Destinations of parcels is in the order give in city_instance_df: parcel 0 -> node 7, parcel 1 -> node 8, etc.
    destinations = [i+total_num_depots_lockers for i in packages]
    
    # Satellite nodes
    locker_nodes = list(city_instance_df.loc[city_instance_df['type']=='locker']['node'])
    
    # Locker capacities
    locker_cap = pd.read_excel('input/locker_capacities.xlsx')
    locker_capacities = dict(zip(locker_cap['Locker no'], locker_cap['Capacity']))
    
    # FM vehicles
    fmd = pd.read_excel('input/first_milers.xlsx')
    num_vehicles_per_FM =list(fmd['num_vehicles'])
    num_FirstMilers = len(fmd['num_vehicles'])
    cost_per_km_for_FM = list(fmd['cost_per_km_for_FM'])
    fm_engine_params = pd.read_excel('input/first_mile_vehicle_engine_params.xlsx')
    
    
    # DSP vehicles
    lmd = pd.read_excel('input/last_milers.xlsx')
    num_vehicles_per_DSP = list(lmd['num_vehicles'])
    num_DSPs = len(lmd['num_vehicles'])
    cost_per_km_for_DSP = list(lmd['cost_per_km_for_DSP'])
    # Battery capacity of electric vehicle in kWh
    bc = pd.read_excel('input/battery_capacity.xlsx')
    battery_capacity = int(bc['Capacity (kWh)'])
    
    # we assume that each DSP has enough capacity. 
    each_dsp_cap = np.ceil((num_packages/num_DSPs))
    dsp_capacities = []
    for i in range(num_DSPs):
        dsp_capacities.append(each_dsp_cap)
        
    
    # DSP depots, nodes and arcs
    last_depots_df = city_instance_df.loc[city_instance_df['type'].str.contains('LM')]
    last_depots_df.reset_index(inplace=True, drop=True)
    dsp_depots = []
    dsp_d_nodes = []
    # Assuming only one depot per DSP
    for d in range(len(last_depots_df)):
        dsp_depots.append(int(last_depots_df['node'][d]))
        dsp_d_nodes.append([int(last_depots_df['node'][d])] + locker_nodes + destinations)
    
    dsp_d_arcs = []
    for d in range(len(dsp_d_nodes)):
        arcs = [(i, j) for i in dsp_d_nodes[d] for j in dsp_d_nodes[d] if i!=j]
        dsp_d_arcs.append(arcs)
        
    
    # FM depots, nodes and arcs
    first_depots_df = city_instance_df.loc[city_instance_df['type'].str.contains('FM')]
    first_depots_df.reset_index(inplace=True, drop=True)
    fm_depots = []
    fm_f_nodes = []
    # Assuming only one depot per FM
    for f in range(len(first_depots_df)):
        fm_depots.append(int(first_depots_df['node'][f]))
        fm_f_nodes.append([int(first_depots_df['node'][f])] + locker_nodes)
    
    fm_f_arcs = []
    for f in range(len(fm_f_nodes)):
        arcs_f = [(i, j) for i in fm_f_nodes[f] for j in fm_f_nodes[f] if i!=j]
        fm_f_arcs.append(arcs_f)
    
    # All nodes
    all_nodes_first_echelon = fm_depots + locker_nodes
    all_nodes_second_echelon = locker_nodes + dsp_depots + destinations
        
    num_nodes = len(city_instance_df)
    distances, travel_times, leave_times, arrive_times = functions.get_distances_and_times(distance_and_times)    
    
    num_time_periods_matrix = functions.get_num_time_periods_matrix(leave_times, num_nodes)
    distance_matrix = functions.get_distance_matrix(distances, num_nodes)
    travel_time_matrix = functions.get_travel_time_matrix(travel_times, num_nodes)
    # final_leave_time = last leave time found in data plus one hour
    final_leave_time = distance_and_times[0]['times']['leave_time'].iloc[-1] + 1 
    leave_time_start, leave_time_end = functions.get_leave_time_matrix(leave_times, num_nodes, final_leave_time)
    earliest, latest = functions.get_time_windows(locations_and_windows)
    bigM_matrix = functions.get_bigM_matrix(arrive_times, latest, num_nodes)
    
    # Assuming average load of 2500kg...
    load = 2500
    emissions_matrix_ICE = functions.compute_emissions_ICE(fm_engine_params, load, distance_matrix, travel_time_matrix, num_time_periods_matrix, num_nodes)
    emissions_matrix_EV = functions.compute_emissions_EV(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, num_time_periods_matrix, num_nodes)
    
    
    time_violation_penalty = 10000
    
    # Problem runtime time limit is also read from file
    tim = pd.read_excel('input/time_limit_per_iteration.xlsx')
    time_limit_per_iteration = int(tim['Time Limit (seconds)'])



    # --- Run Optimizer ---
    t = pc()
    bilevel_HPR_emissions_model = optimization.HPR_model_emissions(time_limit_per_iteration, packages, destinations, locker_nodes, locker_capacities, num_FirstMilers, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, all_nodes_first_echelon,
                            all_nodes_second_echelon, num_time_periods_matrix, emissions_matrix_ICE, emissions_matrix_EV, Pf, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs,
                            travel_time_matrix, bigM_matrix, earliest, latest, leave_time_start, leave_time_end)
    

    
    hpr_sol_final, lamda_sol_final, w_sol_final, xm_sol_final, y_sol_final = optimization.cutting_plane_algorithm(bilevel_HPR_emissions_model, Pf, packages, destinations, locker_nodes, num_nodes, num_FirstMilers, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, cost_per_km_for_FM, cost_per_km_for_DSP, 
                                distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs, 
                                num_time_periods_matrix, leave_time_start, leave_time_end, time_violation_penalty, False)
    
    
    locker_assignments = functions.get_locker_assignments(lamda_sol_final)
    

    # --- Write solution to file ---
    functions.extract_solutions_and_write(xm_sol_final, y_sol_final, locker_assignments, lmd, dsp_depots)
    print(f"Total time taken: {pc()-t}")


if __name__ == "__main__":
    main()