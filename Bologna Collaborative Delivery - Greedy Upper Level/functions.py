# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
import optimization
from geopy.distance import geodesic
pd.options.mode.chained_assignment = None


def generate_distance_time_mat(nodes_df, time_periods_multiplier):
    dist_mat = []
    time_mat = []

    avg_speed = 1250 #12.5 # m/s
    for i in range(len(nodes_df)):
        d_temp = []
        t_temp = []
        origin = (nodes_df.loc[i]["latitude"], nodes_df.loc[i]["longitude"])
        for j in range(len(nodes_df)):
            destination = (nodes_df.loc[j]["latitude"], nodes_df.loc[j]["longitude"])
            d_inner = []
            t_inner = []
            
            for k in time_periods_multiplier:
                dist = round(geodesic(origin, destination).meters * k, 2)
                time = round(dist/avg_speed, 2)
                
                d_inner.append(dist)
                t_inner.append(time)
                
            d_temp.append(d_inner)
            t_temp.append(t_inner)

        dist_mat.append(d_temp)
        time_mat.append(t_temp)
        
    return dist_mat, time_mat

def get_num_time_periods(lst_of_lst_of_lst):
    lengths_result = []
    for inner_lst in lst_of_lst_of_lst:
        inner_lengths = [len(innermost_lst) for innermost_lst in inner_lst]
        lengths_result.append(inner_lengths)
    return lengths_result


def generate_all_times_and_bigMs(city_instance_df, start_time_periods,end_time_periods, travel_time_matrix):
    leave_time_start = []
    leave_time_end = []

    for i in range(len(city_instance_df)):
        leave_start_temp = []
        leave_end_temp = []
        for j in range(len(city_instance_df)):
            leave_start_inner = []
            leave_end_inner = []
            for m in start_time_periods:
                leave_start_inner.append(m)

            for m in end_time_periods:
                leave_end_inner.append(m)

            leave_start_temp.append(leave_start_inner)
            leave_end_temp.append(leave_end_inner)

        leave_time_start.append(leave_start_temp)
        leave_time_end.append(leave_end_temp)

    
    # arrive time = leave_time_start + travel_time
    arrive_time = []
    for i in range(len(leave_time_start)):
        arrive_temp = []

        for j in range(len(leave_time_start[i])):
            arrive_inner = []
            for m in range(len(leave_time_start[i][j])):
                arrive_inner.append(leave_time_start[i][j][m] + travel_time_matrix[i][j][m])

            arrive_temp.append(arrive_inner)

        arrive_time.append(arrive_temp)
    
    # big M for optimization
    bigM_matrix = []
    for i in range(len(arrive_time)):
        big_temp = []
        for j in range(len(arrive_time[i])):
            big_temp.append(max(arrive_time[i][j])+ city_instance_df.iloc[i]['latest'])

        bigM_matrix.append(big_temp)
        
    earliest = list(city_instance_df['earliest'])
    latest = list(city_instance_df['latest'])
    
    return leave_time_start, leave_time_end, arrive_time, earliest, latest, bigM_matrix
    
def compute_distance(all_nodes_df, locker_num, deliv_loc):
    # 1. Get closest lockers to delivery locations
    # This is using basic distance. We can improve on this later...
    origin = (all_nodes_df.loc[locker_num]["latitude"], all_nodes_df.loc[locker_num]["longitude"])
    destination = (all_nodes_df.loc[deliv_loc]["latitude"], all_nodes_df.loc[deliv_loc]["longitude"])
    distance = round(geodesic(origin, destination).meters, 4)
    
    return distance

def get_locker_rankings(all_nodes_df, locker_nodes, delivery_nodes):
    rankings = []
    for i in delivery_nodes:
        distances = {}
        for j in locker_nodes:
            dist = compute_distance(all_nodes_df, j, i)
            distances[j] = dist

        distances = sorted(distances, key = distances.get)
        rankings.append(distances)
    
    return rankings

def get_DSP_rankings(all_nodes_df, dsp_depots, destinations):
    rankings = []
    for i in destinations:
        distances = {}
        for j in dsp_depots:
            dist = compute_distance(all_nodes_df, j, i)
            distances[j] = dist

        distances = sorted(distances, key = distances.get)
        rankings.append(distances)
    
    return rankings

def assign_packages_to_lockers(items, rankings, original_capacities, locker_nodes):
    capacities = [i for i in original_capacities]
    # Create a dictionary to store items for each list
    lists_dict = {i: [] for i in range(len(capacities))}
    
    # Combine items, rankings, and capacities into a list of tuples
    item_data = list(zip(items, rankings))
    
    # Sort items based on rankings in descending order
    sorted_items = sorted(item_data, key=lambda x: x[1])

    # Iterate through sorted items and assign them to lists based on rankings and capacities
    for item, ranking in sorted_items:
        assigned = False
        # Sort lists based on remaining capacity in ascending order to prioritize lists with more space
        for i in sorted(lists_dict, key=lambda x: capacities[x]):
            if capacities[i] > 0:
                lists_dict[i].append(item)
                capacities[i] -= 1
                assigned = True
                break
        if not assigned:
            # If all lists are full, assign the item to the list with the highest ranking
            max_ranking_list = max(lists_dict, key=lambda x: item_data[x][1])
            lists_dict[max_ranking_list].append(item)
            capacities[max_ranking_list] -= 1
    
    # Renaming keys
    lists_dict = dict(zip(locker_nodes, lists_dict.values()))
    
    return lists_dict

def generate_lamda(assignments, num_packages, num_nodes):
    lamda_ps = np.zeros((num_packages, num_nodes))
    for key, indices in assignments.items():
        lamda_ps[indices, key] = 1 
    return lamda_ps

def create_bounds(num_vehicles_per_DSP):
    veh_ranges = np.cumsum(num_vehicles_per_DSP)
    bounds = []
    bounds.append(range(veh_ranges[0]))
    for i in range(len(veh_ranges)):
        if i != 0:
            bounds.append(range(veh_ranges[i-1], veh_ranges[i]))

    return bounds

def generate_assignment_array(input_dict):
    max_key = max(input_dict.keys())
    max_value = max(max(input_dict.values(), key=lambda x: max(x)))

    output_array = np.zeros((max_value + 1, max_key + 1))

    for key, indices in input_dict.items():
        for index in indices:
            output_array[index, key] = 1

    return output_array

def assign_packages_to_DSPs(items, rankings, dsp_capacities):
    capacities = [i for i in dsp_capacities]
    # Create a dictionary to store items for each list
    lists_dict = {i: [] for i in range(len(capacities))}
    
    # Combine items, rankings, and capacities into a list of tuples
    item_data = list(zip(items, rankings))
    
    # Sort items based on rankings in descending order
    sorted_items = sorted(item_data, key=lambda x: x[1])

    # Iterate through sorted items and assign them to lists based on rankings and capacities
    for item, ranking in sorted_items:
        assigned = False
        # Sort lists based on remaining capacity in ascending order to prioritize lists with more space
        for i in sorted(lists_dict, key=lambda x: capacities[x]):
            if capacities[i] > 0:
                lists_dict[i].append(item)
                capacities[i] -= 1
                assigned = True
                break
        if not assigned:
            # If all lists are full, assign the item to the list with the highest ranking
            max_ranking_list = max(lists_dict, key=lambda x: item_data[x][1])
            lists_dict[max_ranking_list].append(item)
            capacities[max_ranking_list] -= 1
    
    dsp_ids = list(range(len(dsp_capacities)))
    # Renaming keys
    lists_dict = dict(zip(dsp_ids, lists_dict.values()))
    
    
    assignments = generate_assignment_array(lists_dict)
    
    return assignments

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

def extract_xm(hpr_sol, emissions_matrix_EV):
    xm_rows = hpr_sol[hpr_sol['name'].str.startswith('xm')]
    xm_rows = xm_rows.reset_index(drop=True)
   
    xm_df = pd.DataFrame(columns=['i','j','k','m','Emissions'])
    for i in range(xm_rows.shape[0]):
        row = xm_rows['name'][i].split('_')
        row.pop(0);    
        row = [int(i) for i in row]
        row.append(emissions_matrix_EV[row[0]][row[1]][row[3]])
        xm_df.loc[len(xm_df)] = row
    
    return xm_df

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
            
def write_locker_assignments_to_file(locker_assignments, file_path):
    with open(file_path, 'w') as file:
        for locker, package_ids in locker_assignments.items():
            file.write(f"Packages assigned to locker {locker}:\n")
            for package_id in package_ids:
                file.write(f"package_id {package_id}\n")
            file.write("\n")
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
        # If solution exists:
        if len(last_m_sol_dfs_final[d]) > 0:
            # Compute total distance by last-miler
            xm_sol_final_d = optimization.extract_xm(last_m_sol_dfs_final[d], num_vehicles_per_DSP, distance_matrix)    
            dist_res.append(xm_sol_final_d['c_ijm'].sum()/1000)  

            # Compute total emissions by last-miler
            total_emissions_d = 0
            for i in range(len(xm_sol_final_d)):
                i_index = int(xm_sol_final_d.loc[i]['i'])
                j_index = int(xm_sol_final_d.loc[i]['j'])
                m_index = int(xm_sol_final_d.loc[i]['m'])
                total_emissions_d += emissions_matrix_EV[i_index][j_index][m_index]
            emm_res.append(total_emissions_d/1000)

        else:
            dist_res.append(0) 
            emm_res.append(0)

    # Create columns based on number of last-milers
    cols = []
    for i in range(len(last_m_sol_dfs_final)):
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
