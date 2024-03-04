# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:42:46 2024

@author: ade.fajemisin
"""


import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import optimization


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


def get_lastmiler_assgt(y_sol_final):
    y_sol_final = np.round(y_sol_final)
    indices = []
    for row in y_sol_final:
        index = next((i for i, x in enumerate(row) if x == 1), None)
        indices.append(index if index is not None else -1)
    return indices

def get_times_at_destinations(last_m_sol_dfs_final, y_sol_final, packages, destinations, dsp_depots, locker_nodes):    
    all_arrive_times = pd.DataFrame(columns=['k','i','time'])
    irrelevant_rows = dsp_depots + locker_nodes
    
    for d in range(len(last_m_sol_dfs_final)):    
        # If solution for DSP d exists:
        if len(last_m_sol_dfs_final[d]) > 0:
            arrive_time_d = optimization.extract_t(last_m_sol_dfs_final[d])
            arrive_time_d_filtered = arrive_time_d[~arrive_time_d['i'].isin(irrelevant_rows)]
            
            all_arrive_times = pd.concat([all_arrive_times, arrive_time_d_filtered], axis=0)
            
    # Get times at which the nodes are visited in HH:MM
    converted_hours = []
    for i in range(len(all_arrive_times)):
        hours = int(all_arrive_times.iloc[i]['time'])
        minutes = int(all_arrive_times.iloc[i]['time']*60) % 60
        converted_hours.append("%02d:%02d" % (hours, minutes))
    
    # Create dataframe and return
    times_at_dest_df = pd.DataFrame()
    times_at_dest_df['Package ID'] = packages
    times_at_dest_df['Destination node'] = destinations
    times_at_dest_df['Arrival Time at Destination (HH:MM)'] = converted_hours
    times_at_dest_df['Carried by Last Miler'] = get_lastmiler_assgt(y_sol_final)
    
    return times_at_dest_df

def get_distance_and_emissions(last_m_sol_dfs_final, num_vehicles_per_DSP, distance_matrix, emissions_matrix_EV):
    dist_res = []
    emm_res = []
    for d in range(len(last_m_sol_dfs_final)): 
        if len(last_m_sol_dfs_final[d]) > 0:
            # Compute total distance by last-miler
            xm_sol_final_d = optimization.extract_xm(last_m_sol_dfs_final[d], num_vehicles_per_DSP, distance_matrix)
#             xm_sol_final_d = xm_sol_final.loc[xm_sol_final['d'] == d]    
            dist_res.append(xm_sol_final_d['c_ijm'].sum())  

            # Compute total emissions by last-miler
            total_emissions_d = 0
            for i in range(len(xm_sol_final_d)):
                i_index = int(xm_sol_final_d.loc[i]['i'])
                j_index = int(xm_sol_final_d.loc[i]['j'])
                m_index = int(xm_sol_final_d.loc[i]['m'])
                total_emissions_d += emissions_matrix_EV[i_index][j_index][m_index]
            emm_res.append(total_emissions_d)
        else:
            dist_res.append(0) 
            emm_res.append(0)
     

    # Create columns based on number of last-milers
    cols = []
    for i in range(len(num_vehicles_per_DSP)):
        cols.append('Last Miler ' + str(i))

    # create dataframe and write to file
    dist_emm_df = pd.DataFrame([dist_res, emm_res], columns = cols, index = (['Total distance (km)', 'Total CO2 emissions (gCO2eq)']))
    
    return dist_emm_df

def write_instance_results_to_file(res_filename, y_sol_final, locker_assignments, last_m_sol_dfs_final, num_vehicles_per_DSP, 
                                   distance_matrix, emissions_matrix_EV, dsp_depots, locker_nodes, packages, destinations):
    
    
    # Get distance travelled and total last-mile emissions
    dist_emm_df = get_distance_and_emissions(last_m_sol_dfs_final, num_vehicles_per_DSP, distance_matrix, emissions_matrix_EV)
      
    # Get package assignments to Lockers
    locker_ass_df = pd.DataFrame([(k, v) for k, vals in locker_assignments.items() for v in vals], columns=['Locker node', 'Package ID'])
    
    # Get package arrival times
    times_at_dest_df = get_times_at_destinations(last_m_sol_dfs_final, y_sol_final, packages, destinations, dsp_depots, locker_nodes)
    
    # Write all to file
    with pd.ExcelWriter(res_filename) as writer:  
        dist_emm_df.to_excel(writer, sheet_name='Distance and Emissions')
        locker_ass_df.to_excel(writer, sheet_name='Assignments to Lockers', index=False)
        times_at_dest_df.to_excel(writer, sheet_name='Package Arrival Times', index=False)

    return