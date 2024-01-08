# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:38:52 2023

@author: ade.fajemisin
"""
import pandas as pd
import functions

def main():
    
    # Inputs
    LMAD_nodes_df = pd.read_excel('input/Helsinki_LMAD_delivery_nodes_8_12_2023.xlsx')
    LMAD_routes = functions.read_json('input/Helsinki_LMAD_robot_routes_8_12_2023.json')
    input_data = pd.read_excel('input/test_input_1.xlsx')
    
    earliest = input_data['earliest']
    latest = input_data['latest']
    wait_times = input_data['wait time (hours)']
    
    
    # Routing
    # num_robots = 1            
    all_arcs_length_time = functions.create_all_arcs(LMAD_nodes_df, LMAD_routes)
    distance_matrix, time_matrix = functions.create_distance_and_time_matrices(all_arcs_length_time, LMAD_nodes_df)
    routing_results = functions.run_scenario(LMAD_nodes_df, distance_matrix, time_matrix, earliest, latest, wait_times)
    
    # Output
    routing_results.to_excel('output/routing_results.xlsx', index = False)
    

    
if __name__ == "__main__":
    main()
