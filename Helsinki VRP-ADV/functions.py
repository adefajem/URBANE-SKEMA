# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:50:22 2023

@author: ade.fajemisin
"""

import json
import pandas as pd
from docplex.mp.model import Model
import itertools 

def read_json(filename):
    with open(filename) as f:
        return json.load(f)
    
def create_all_arcs(LMAD_nodes_df, LMAD_routes):    
    # We don't need the crossing nodes to simulate the robots routes, so only keep links that don't have
    # "Crossing" in them
    delivery_arcs = []

    for i in range(len(LMAD_routes["features"])):    
        if "Crossing" not in LMAD_routes["features"][i]["properties"]["name"]:
            delivery_arcs.append(LMAD_routes["features"][i])

    # Create list of dictionaries, where each dictionary has the keys 'from', 'to', 'length', 'time'
    from_list = []
    to_list = []
    for i in range(len(delivery_arcs)):
        a, b = delivery_arcs[i]["properties"]["name"].split("=")
        from_list.append(a)
        to_list.append(b)

    # This is the current data we have
    time_distance_list = [] 
    for i in range(len(delivery_arcs)):
        temp_dic = {}
        temp_dic["from"] = from_list[i]
        temp_dic["to"] = to_list[i]
        temp_dic["length"] = delivery_arcs[i]["properties"]["length"]
        temp_dic["time"] = delivery_arcs[i]["properties"]["time"]
        time_distance_list.append(temp_dic)

    
    # The above data is incomplete, i.e. the graph isn't fully connected, so we make it fully connected
    nodes = list(LMAD_nodes_df["name"])
    arc_combos = list(itertools.product(nodes, repeat=2))
    longest_arc = max(time_distance_list, key = lambda x:x["length"])
    
    all_arcs = []
    for i in range(len(arc_combos)):
        temp_dic = {}
        temp_dic['from'] = arc_combos[i][0]
        temp_dic['to'] = arc_combos[i][1]
        temp_dic['length'] = longest_arc["length"]
        temp_dic['time'] = longest_arc["time"]/5
        
        all_arcs.append(temp_dic)

    for i in range(len(all_arcs)):
        for j in range(len(time_distance_list)):
            if all_arcs[i]['from'] == time_distance_list[j]['from'] and all_arcs[i]['to'] == time_distance_list[j]['to']:
                all_arcs[i]['length'] = time_distance_list[j]['length']
                all_arcs[i]['time'] = time_distance_list[j]['time']

    # Change node names to numbers
    name_to_num_match = dict(zip(LMAD_nodes_df["name"], LMAD_nodes_df["id_num"]))
    
    for i in range(len(all_arcs)):
        for j in range(len(name_to_num_match)):
            if all_arcs[i]['from'] == list(name_to_num_match.keys())[j]:
                all_arcs[i]['from'] = list(name_to_num_match.values())[j]
            if all_arcs[i]['to'] == list(name_to_num_match.keys())[j]:
                all_arcs[i]['to'] = list(name_to_num_match.values())[j]

    return all_arcs

def create_distance_and_time_matrices(all_arcs_length_time, LMAD_nodes_df):    
    lengths = []
    times = []

    for i in range(len(all_arcs_length_time)):
        lengths.append(all_arcs_length_time[i]['length'])
        times.append(all_arcs_length_time[i]['time']/3600) # Convert to hours
    
    # Create distance and time matrix
    num_nodes = len(LMAD_nodes_df)
    distance_matrix = [lengths[i:i + num_nodes] for i in range(0, len(lengths), num_nodes)]
    time_matrix = [times[i:i + num_nodes] for i in range(0, len(times), num_nodes)]
    
    return distance_matrix, time_matrix

def generate_bigM_matrix(earliest, latest, wait_time, time_matrix):
    bigM_matrix = []
    
    for i in range(len(time_matrix)):
        temp = []
        for j in range(len(time_matrix[i])):
            temp.append(latest[i] + wait_time[i] + time_matrix[i][j])
        bigM_matrix.append(temp)   
    
    return bigM_matrix

def robot_routing(distance_matrix, time_matrix, bigM_matrix, earliest, latest, wait_times, num_nodes):
    model = Model(name = 'LMAD_routing')
    model.parameters.timelimit = 30
    # ----- Sets -----
    K = range(1) #range(num_robots)
    V = range(num_nodes)  
    A = [(i,j) for i in V for j in V if i!=j]
        
    # ----- Variables ----- 
    # x_kij = 1 if arc (i,j) is traversed by vehicle k  
    x = {(k,i,j):model.binary_var(name='x_%d_%d_%d' % (k,i,j)) for k in K for i in V for j in V if i!=j}
    
    # Arrival time of vehicle k at node i - s_ki
    s = {(k,i):model.continuous_var(lb=0.0, name='s_%d_%d' % (k,i)) for k in K for i in V}
    
    # ----- Objective function ----- 
    model.minimize(model.sum(distance_matrix[i][j] * x[k,i,j] for k in K for (i,j) in A))
         

    # ----- Constraints -----
    # Leave depot
    for k in K:
        model.add_constraint(model.sum(x[k,0,j] for j in V if j!=0) == 1, ctname = "leave depot")
    
   
    # Visit all delivery nodes
    for i in V:
         model.add_constraint(model.sum(x[k,i,j] for k in K for j in V if i!=j) == 1, ctname = "visit all nodes")
    
    
    # Flow conservation
    for k in K:
        for j in V:    
            model.add_constraint(model.sum(x[k,i,j] for i in V if i!=j) - model.sum(x[k,j,i] for i in V if i!=j) == 0,
                                ctname = "flow conservation")
        
    # Arrival times
    for (i,j) in A:
        if j==0: continue
        for k in K:
            model.add_constraint(s[k,i] + wait_times[i] + time_matrix[i][j] <= s[k,j] + (1 - x[k,i,j]) * bigM_matrix[i][j],
                                ctname = "arrival times")
         
    # Respect time windows
    for k in K:
        for i in V:
            model.add_constraint(earliest[i] <= s[k,i], ctname = "early window")
            model.add_constraint(s[k,i] <= latest[i], ctname = "late window")
         
         
    # Solve
    solution = model.solve(log_output = False)   
#     print(model.solve_status.name)
    obj_val = None
    sol_df = None
    

    if model.solve_status.name == 'OPTIMAL_SOLUTION' or model.solve_status.name == 'FEASIBLE_SOLUTION':
        obj_val = solution.get_objective_value()
        sol_df = solution.as_df()
    
    return sol_df

def output_result(route_sol, LMAD_nodes_df):    
    result = None
    
    try:
        # Extract times
        t_rows = route_sol[route_sol['name'].str.startswith('s')]    
        t_rows = t_rows.sort_values("value")
        t_rows = t_rows.reset_index(drop=True)

        # Get times at which the nodes are visited in HH:MM
        t_df = pd.DataFrame(columns=['j','Arrival time'])
        for i in range(t_rows.shape[0]):
            row = t_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            j = row[-1]
            hours = int(t_rows['value'][i])
            minutes = int(t_rows['value'][i]*60) % 60
            res = [j,"%02d:%02d" % (hours, minutes)]
            t_df.loc[len(t_df)] = res

        # Match node numbers to names
        name_to_num_match = dict(zip(LMAD_nodes_df["name"], LMAD_nodes_df["id_num"]))

        names =[]
        for i in range(len(t_df)):
            for j in range(len(name_to_num_match)):
                if t_df.loc[i]["j"] == list(name_to_num_match.values())[j]:
                    names.append(list(name_to_num_match.keys())[j])

        t_df["Delivery point"] = names
        result = t_df[['Delivery point','Arrival time']].copy()    
        
    except:
        print('The problem is infeasible. Ajust inputs and try again.')
    
    return result

def run_scenario(LMAD_nodes_df, distance_matrix, time_matrix, earliest, latest, wait_times):
    
    num_nodes = len(LMAD_nodes_df)
    # Routing
    bigM_matrix = generate_bigM_matrix(earliest, latest, wait_times, time_matrix)
    route_sol = robot_routing(distance_matrix, time_matrix, bigM_matrix, earliest, latest, wait_times, num_nodes)

    res = output_result(route_sol, LMAD_nodes_df)
    
    return res