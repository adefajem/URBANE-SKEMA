# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:42:46 2024

@author: ade.fajemisin
"""


import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np


import networkx as nx

# Get leave_times, arrive_times, distances and travel_times into 'matrix' form
def get_distances_and_times(distance_and_times):
    distances = []
    travel_times = []
    leave_times = []
    arrive_times = []

    for i in range(len(distance_and_times)):
        a = distance_and_times[i]['from']
        b = distance_and_times[i]['to']

        arc_distances_dict = {'from': int(a), 'to': int(b), 'distances': distance_and_times[i]['times']['distance'].to_list()}
        distances.append(arc_distances_dict)

        arc_travel_times_dict = {'from': int(a), 'to': int(b), 'travel_times': distance_and_times[i]['times']['travel_time'].to_list()}
        travel_times.append(arc_travel_times_dict)

        arc_leave_times_dict = {'from': int(a), 'to': int(b), 'leave_times': distance_and_times[i]['times']['leave_time'].to_list()}
        leave_times.append(arc_leave_times_dict)

        arc_arrive_times_dict = {'from': int(a), 'to': int(b), 'arrive_times': distance_and_times[i]['times']['arrive_time'].to_list()}
        arrive_times.append(arc_arrive_times_dict)

    # Sort so that array goes from 0 to numNodes
    distances = sorted(distances, key=lambda k: (k.get('from', 0),k.get('to', 0)))
    travel_times = sorted(travel_times, key=lambda k: (k.get('from', 0),k.get('to', 0)))
    leave_times = sorted(leave_times, key=lambda k: (k.get('from', 0),k.get('to', 0)))
    arrive_times = sorted(arrive_times, key=lambda k: (k.get('from', 0),k.get('to', 0)))
    
    
    return distances, travel_times, leave_times, arrive_times
    
def get_num_time_periods_matrix(instance_tt_leave_times, numNodes): 
    # Find the number of time periods for each arc
    num_time_periods = []
    for i in range(len(instance_tt_leave_times)):
        num_time_periods.append(len(instance_tt_leave_times[i]['leave_times']))

    num_time_periods_matrix = [num_time_periods[i:i + numNodes] for i in range(0, len(num_time_periods), numNodes)]
    
    return num_time_periods_matrix

def get_bigM_matrix(instance_tt_arrive_times, latest, numNodes):    
    # Put in matrix from
    max_arrive_time_per_arc = []
    for i in range(len(instance_tt_arrive_times)):
        max_arrive_time_per_arc.append(max(instance_tt_arrive_times[i]['arrive_times']))

    bigM_matrix = [max_arrive_time_per_arc[i:i + numNodes] for i in range(0, len(max_arrive_time_per_arc), numNodes)]
    
    for i in range(len(bigM_matrix)):
        for j in range(len(bigM_matrix[i])):
            bigM_matrix[i][j] = bigM_matrix[i][j] + latest[i]
    
    return bigM_matrix

def get_distance_matrix(instance_tt_distances, numNodes):   
    # Put in matrix from
    all_distances = []
    for i in range(len(instance_tt_distances)):
#         all_distances.append(instance_tt_distances[i]['distances'])
        all_distances.append(list(np.array(instance_tt_distances[i]['distances'])/1000)) # convert to kilometers

    distance_matrix = [all_distances[i:i + numNodes] for i in range(0, len(all_distances), numNodes)]
    
    return distance_matrix

def get_travel_time_matrix(instance_tt_travel_times, numNodes):    
    # Put in matrix from
    all_travel_times = []
    for i in range(len(instance_tt_travel_times)):
        all_travel_times.append(instance_tt_travel_times[i]['travel_times'])

    travel_time_matrix = [all_travel_times[i:i + numNodes] for i in range(0, len(all_travel_times), numNodes)]
    
    return travel_time_matrix

def get_leave_time_matrix(instance_tt_leave_times, numNodes, final_leave_time):  
    # Put in matrix from
    all_leave_times = []
    all_leave_times_shifted = []
    for i in range(len(instance_tt_leave_times)):
        all_leave_times.append(instance_tt_leave_times[i]['leave_times'])
        
        # shifted so that we can have the upper bound on leave times: b_ijm+1
        shifted = instance_tt_leave_times[i]['leave_times'][1:]
        shifted.append(final_leave_time)
        all_leave_times_shifted.append(shifted)

    leave_time_start = [all_leave_times[i:i + numNodes] for i in range(0, len(all_leave_times), numNodes)]
    leave_time_end = [all_leave_times_shifted[i:i + numNodes] for i in range(0, len(all_leave_times_shifted), numNodes)]
    
    return leave_time_start, leave_time_end
    
def get_time_windows(time_windows_data):
    earliest = []
    latest = []
    toHours = 3600000
    
    for i in range(len(time_windows_data)):
        earliest.append(time_windows_data[i]['earliest_delivery'])
        latest.append(time_windows_data[i]['latest_delivery'])
    
    earliest = list(np.array(earliest)/toHours)
    latest = list(np.array(latest)/toHours)
    
    # Prepend time window for depot
    earliest = [min(earliest)] + earliest
    latest = [max(latest)] + latest
    
    return earliest, latest

def compute_emissions_ICE(fm_engine_params, load, distance_matrix, travel_time_matrix, num_time_periods_matrix, num_nodes):    
    
    k_e = float(fm_engine_params['engine friction factor'])
    N_e = float(fm_engine_params['engine speed'])
    V_e = float(fm_engine_params['engine displacement'])
    alpha = float(fm_engine_params['alpha']) # alpha, beta, gamma and lambda are constants from Franceschetti et al. (2013)
    beta = float(fm_engine_params['beta'])
    gamma = float(fm_engine_params['gamma'])
    lamda = float(fm_engine_params['lambda'])
    mu_eng = float(fm_engine_params['curb weight (in kg)'])
    
    emission_ij = []
    
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):  
            emissions_m = []
            for m in range(num_time_periods_matrix[i][j]): 
                
                if travel_time_matrix[i][j][m]!=0:
                    emission = lamda * (k_e * N_e * V_e * travel_time_matrix[i][j][m] \
                                        + gamma * beta * distance_matrix[i][j][m] * (distance_matrix[i][j][m] / travel_time_matrix[i][j][m])**2 \
                                                   + gamma * alpha * (mu_eng + load) * distance_matrix[i][j][m])                    
                    emissions_m.append(emission*20e4)  
                else:
                    emissions_m.append(0)
            
            emission_ij.append(emissions_m)
    
    # Put in matrix form
    E_ijkm = [emission_ij[i:i+num_nodes] for i in range(0, len(emission_ij), num_nodes)]
    
    return E_ijkm


def EVCO2(kwh_consumption, emission_factor, generation_percentage):
    energy_mix = sum([emission_factor[i] * generation_percentage[i] for i in range(len(emission_factor))])
    
    # if you multiply gCO2eq/kWh with kWh you get gCO2eq
    emissions_gCO2 = kwh_consumption * energy_mix
    
    return emissions_gCO2 # in gCO2eq
    

def compute_emissions_EV(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, num_time_periods_matrix, num_nodes):
    emission_ij = []
    
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):  
            emissions_m = []
            for m in range(num_time_periods_matrix[i][j]): 
                if travel_time_matrix[i][j][m]!=0:
                    kwh_consumption = battery_capacity * distance_matrix[i][j][m] / 100 # kilowatt-hours/100 km
                    emission = EVCO2(kwh_consumption, emission_factor, generation_percentage)
                                  
                    emissions_m.append(emission)
                else:
                    emissions_m.append(0)
            
            emission_ij.append(emissions_m)
                    
    # Put in matrix form
    E_ijkm = [emission_ij[i:i+num_nodes] for i in range(0, len(emission_ij), num_nodes)]
                    
    return E_ijkm

def create_bounds(num_vehicles_per_DSP):
    veh_ranges = np.cumsum(num_vehicles_per_DSP)
    bounds = []
    bounds.append(range(veh_ranges[0]))
    for i in range(len(veh_ranges)):
        if i != 0:
            bounds.append(range(veh_ranges[i-1], veh_ranges[i]))

    return bounds

def get_firstmiler_package_origins(data):
    groups = {}
    
    for idx, origin in enumerate(data['first_miler_of_origin']):
        if origin not in groups:
            groups[origin] = []
        groups[origin].append(idx)
    
    return groups


def write_assignments_to_file(y_col, d, assignment_file_path):
    with open(assignment_file_path, 'w') as file:       
        # Find indices where the value is 1
        indices = np.where(y_col == 1)[0]
        
        # Generate the output string
        output_string = "\n".join([f"package_id {index}" for index in indices])

        # Write the output to a file
        with open(assignment_file_path, 'w') as file:
            file.write("Packages assigned to last-miler " + str(d) +":\n")
            file.write(output_string)
    return

def get_locker_assignments(lamda_final):
    assignments = {}

    # Iterate over columns
    for col in range(lamda_final.shape[1]):
        # Get the indices where the value is 1
        indices = np.where(lamda_final[:, col] == 1)[0]

        # If there are indices, add them to the assignments dictionary
        if len(indices) > 0:
            assignments[col] = indices.tolist()
            
    return assignments

def write_locker_assignments_to_file(locker_assignments, file_path):
    with open(file_path, 'w') as file:
        for locker, package_ids in locker_assignments.items():
            file.write(f"Packages assigned to locker {locker}:\n")
            for package_id in package_ids:
                file.write(f"package_id {package_id}\n")
            file.write("\n")
    return

def add_dsp_depot_column(df1, df2):
    # Assuming 'd' column is present in df2
    df2['dsp_depot'] = df2['d'].apply(lambda x: df1['LM_id'][0] if x == 0 else df1['LM_id'][1])
    return df2

def get_path(xm_sol_final, d):
    # Create a directed graph from the DataFrame
    graph = nx.from_pandas_edgelist(xm_sol_final, 'i', 'j', create_using=nx.DiGraph())    
    # Perform depth-first search to find the path
    path = nx.dfs_preorder_nodes(graph, source=d)    
    path = list(path)
    
    return list(map(int, path))

def write_dsp_path_to_file(my_list, file_path):
    with open(file_path, 'w') as file:
        file.write(" -> ".join(map(str, my_list)))
    return

def extract_solutions_and_write(xm_sol_final, y, locker_assignments, lmd, dsp_depots):  
    print('Writing solutions to file...')
    
    # Write package assignments to Lockers
    locker_file_path = 'output/Package_assignments_to_lockers.txt'
    write_locker_assignments_to_file(locker_assignments, locker_file_path)
    
    # Write DSP path to file
    xm_sol_final = add_dsp_depot_column(lmd, xm_sol_final)
    for i in dsp_depots:
        xm_d = xm_sol_final[xm_sol_final['dsp_depot'] == i]    
        if len(xm_d) != 0:
            # Get path
            dsp_path = get_path(xm_d, i)
            # Write to file
            dd = lmd.index[lmd['LM_id'] == i].to_list()[0]
            file_path = 'output/last-miler_'+ str(dd) + '_path.txt'
            write_dsp_path_to_file(dsp_path, file_path)

    for d in range(y.shape[1]):
        # Write package assignments to DSPs
        assignment_file_path = 'output/Package_assignments_to_last-miler_'+ str(d) + '.txt'
        write_assignments_to_file(y[:, d], d, assignment_file_path)
        
    print('Done!')

    return
