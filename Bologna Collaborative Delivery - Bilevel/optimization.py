# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:43:12 2024

@author: ade.fajemisin
"""
from docplex.mp.model import Model
import pandas as pd
import numpy as np
import functions


def HPR_model_emissions(timelimitSecs, packages, destinations, locker_nodes, locker_capacities, num_FirstMilers, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, all_nodes_first_echelon,
                        all_nodes_second_echelon, num_time_periods_matrix, emissions_matrix_ICE, emissions_matrix_EV, Pf, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs,
                        travel_time_matrix, bigM_matrix, earliest, latest, leave_time_start, leave_time_end):
    model = Model(name = 'HPR')
    model.parameters.timelimit = timelimitSecs
    
    # ----- Sets -----
    P = packages 
    satellites = locker_nodes
    F = range(num_FirstMilers)
    D = range(num_DSPs)
    
    total_num_vehicles_First_Mile = sum(num_vehicles_per_FM)
    KF = range(total_num_vehicles_First_Mile) # The set of all vehicles in the first-echelon
    
    total_num_vehicles_Last_Mile = sum(num_vehicles_per_DSP)
    KD = range(total_num_vehicles_Last_Mile)
    
    V1 = all_nodes_first_echelon
    A1 = [(i,j) for i in V1 for j in V1 if i!=j]
    
    V2 = all_nodes_second_echelon
    A2 = [(i,j) for i in V2 for j in V2 if i!=j]
    
    
    M = num_time_periods_matrix
    
    
    vehicle_bounds_FMs = functions.create_bounds(num_vehicles_per_FM)
    vehicle_bounds_DSPs = functions.create_bounds(num_vehicles_per_DSP)
    
    
    # ----- Variables -----
    # ----- Leader Variables -----
    # y_pd = 1 if parcel p is offered to DSP d by the leader
    y = {(p,d):model.binary_var(name='y_%d_%d' % (p,d)) for p in P for d in D}
    
    # lamda_ps = 1 if parcel p is placed at satellite s
    lamda = {(p,s):model.binary_var(name='lamda_%d_%d' % (p,s)) for p in P for s in satellites}
    
    # ----- First-Mile Variables -----
    # w_kij = 1 if arc (i,j) is traversed by vehicle k  
    w = {(k,i,j):model.binary_var(name='w_%d_%d_%d' % (k,i,j)) for k in KF for i in V1 for j in V1 if i!=j}
        
    # Arrival time of vehicle k at node i - tau_ki
    tau = {(k,i):model.continuous_var(lb=0.0, name='tau_%d_%d' % (k,i)) for k in KF for i in V1} 
    
    # ----- Last-Mile Variables -----
    # mu_pk = 1 if package p is caried by vehicle k
    mu = {(p,k):model.binary_var(name='mu_%d_%d' % (p,k)) for p in P for k in KD }
    
    # x_kij = 1 if arc (i,j) is traversed by vehicle k  
    x = {(k,i,j):model.binary_var(name='x_%d_%d_%d' % (k,i,j)) for k in KD for i in V2 for j in V2 if i!=j}
    
    # x_kmij = 1 if arc (i,j) is traversed by vehicle k in time period m     
    x_m = {(i,j,k,m):model.binary_var(name='xm_%d_%d_%d_%d' % (i,j,k,m)) for i in V2
                                     for j in V2 if j!=i
                                     for k in KD
                                     for m in range(M[i][j]) }
    
    # Arrival time of vehicle k at node i - t_ki
    t = {(k,i):model.continuous_var(lb=0.0, name='t_%d_%d' % (k,i)) for k in KD for i in V2}  
    
    # Variable theta used to linearize the non-convex quadratic constraint
    theta = {(k,p,s):model.continuous_var(lb=0.0, name='theta_%d_%d_%d' % (k,p,s)) for k in KD for p in P for s in satellites}
    
    # Variables for earliest and latest times
    alpha_early = {(k,i): model.continuous_var(lb=0.0, name='alphaEarly_%d_%d' % (k,i)) for k in KD for i in V2}
    alpha_late = {(k,i): model.continuous_var(lb=0.0, name='alphaLate_%d_%d' % (k,i)) for k in KD for i in V2}
    
    
    # ----- Leader Objective Function -----
    cost_per_DSP_vehicle = 500
    # Minimize emissions
    model.minimize(model.sum(cost_per_DSP_vehicle * x_m[i,j,k,m] for (i,j) in A2 for k in KD for m in range(M[i][j]))
                   + model.sum(emissions_matrix_ICE[i][j][0] * w[k,i,j] for (i,j) in A1 for k in KF)
                   + model.sum(emissions_matrix_EV[i][j][m] * x_m[i,j,k,m] for (i,j) in A2 for k in KD for m in range(M[i][j]))
                  ) 
    
      
    # ----- Leader Constraints -----
    # Respect satellites' capacity constraint
    for s in satellites:
        model.add_constraint(model.sum(lamda[p,s] for p in P) <= locker_capacities[s])
    
    # A parcel should only be assigned to one satellite
    for p in P:
        model.add_constraint(model.sum(lamda[p,s] for s in satellites) == 1)    
    
    # Only one DSP should be assigned to each parcel
    for p in P:
        model.add_constraint(model.sum(y[p,d] for d in D) == 1)    
        
        
    # ----- First-Mile Follower Constraints -----
    # Link assignment and routing variables
    for f in F:
        for p in Pf[f]:
            for s in satellites:
                model.add_constraint(model.sum(w[k,i,s] for i in fm_f_nodes[f] for k in vehicle_bounds_FMs[f] if i!=s) >= lamda[p,s])
        
    # Leave depot
    for f in F:
        for k in vehicle_bounds_FMs[f]:
            model.add_constraint(model.sum(w[k,fm_depots[f],s] for s in satellites) <= 1)
        
    # Flow conservation in first echelon
    for f in F:
        for i in fm_f_nodes[f]:
            for k in vehicle_bounds_FMs[f]:
                model.add_constraint(model.sum(w[k,j,i] for j in fm_f_nodes[f] if i!=j) - model.sum(w[k,i,j] for j in fm_f_nodes[f] if i!=j) == 0)  
        
    # Arrival time
    for f in F:
        for (i,j) in fm_f_arcs[f]:
            if j==fm_depots[f]: continue
            for k in vehicle_bounds_FMs[f]:
                model.add_constraint(tau[k,i] + travel_time_matrix[i][j][0] <= 
                                     tau[k,j] + (1 - w[k,i,j]) * bigM_matrix[i][j])
        
    # Earliest and latest time windows
    for f in F:
        for k in vehicle_bounds_FMs[f]:
            for i in fm_f_nodes[f]:
                model.add_constraint(earliest[i]*model.sum(w[k,i,j] for j in fm_f_nodes[f] if i!=j) <= tau[k,i])
                model.add_constraint(tau[k,i] <= latest[i]*model.sum(w[k,i,j] for j in fm_f_nodes[f] if i!=j))
    
    
    
    # ----- Last-Mile Follower Constraints -----
    # Time Violation Constraints
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            for i in dsp_d_nodes[d]:
                model.add_constraint(alpha_early[k,i] >= earliest[i]*model.sum(x[k,i,j] for j in dsp_d_nodes[d] if i!=j) - t[k,i])
                model.add_constraint(alpha_late[k,i] >= t[k,i] - latest[i]*model.sum(x[k,i,j] for j in dsp_d_nodes[d] if i!=j))



    # If a DSP is assigned a package, it should be carried on one vehicle k
    for d in D:
        for p in P:
            model.add_constraint(model.sum(mu[p,k] for k in vehicle_bounds_DSPs[d]) == y[p,d])
        
    # If a vehicle k picks up a package p, it should go to the destination of p
    for d in D:
        for p in P:
            for k in vehicle_bounds_DSPs[d]:
                model.add_constraint(model.sum(x[k,j,destinations[p]] for j in dsp_d_nodes[d] if j != destinations[p]) == mu[p,k])
    
    # A package p should be picked up from its satellite s
    for d in D:
        for p in P:
            for k in vehicle_bounds_DSPs[d]:
                model.add_constraint(model.sum(lamda[p,s]*x[k,s,j] for s in satellites for j in dsp_d_nodes[d] if s!=j) >= mu[p,k])
    
    # A vehicle shoud go from a depot to a satellite
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            model.add_constraint(model.sum(x[k,dsp_depots[d],s] for s in satellites) <= model.sum(mu[p,k] for p in P))
    
    # A vehicle may go from a satellite to another node
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            for s in satellites:
                model.add_constraint(model.sum(x[k,s,j] for j in dsp_d_nodes[d] if s!=j) <= 1)
    
    # Flow conservation at all nodes in second echelon
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            for j in dsp_d_nodes[d]:    
                model.add_constraint(model.sum(x[k,i,j] for i in dsp_d_nodes[d] if i!=j) - model.sum(x[k,j,i] for i in dsp_d_nodes[d] if i!=j) == 0)
 
    
    # Satellite Time constraints - Non-convex quadratic constraint that has been linearized
    for p in P:
        for d in D:
            for k in vehicle_bounds_DSPs[d]:
                model.add_constraint(model.sum(theta[k,p,s] + lamda[p,s]*min(travel_time_matrix[s][destinations[p]]) for s in satellites) <= 
                                     t[k, destinations[p]] + (1-mu[p,k])*max(max(bigM_matrix)))#bigN[p][k])
    
    for p in P:
        for d in D:
            for k in vehicle_bounds_DSPs[d]:
                for s in satellites:
                    model.add_constraint(theta[k,p,s] <= lamda[p,s] * max(max(bigM_matrix)))
                    model.add_constraint(theta[k,p,s] <= t[k,s] * max(max(bigM_matrix)))
                    model.add_constraint(theta[k,p,s] >= t[k,s] - (1-lamda[p,s])*max(max(bigM_matrix)))

    
    # Arrival time
    for d in D:
        for (i,j) in dsp_d_arcs[d]:
            if j==dsp_depots[d]: continue
            for k in vehicle_bounds_DSPs[d]:
                model.add_constraint(t[k,i] + model.sum(x_m[i,j,k,m] * travel_time_matrix[i][j][m] for m in range(M[i][j]) if i!=j) <= 
                                     t[k,j] + (1 - x[k,i,j]) * bigM_matrix[i][j])
    
    # Time period
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            for i in dsp_d_nodes[d]:
                model.add_constraint(model.sum(x_m[i,j,k,m]*leave_time_start[i][j][m] for j in dsp_d_nodes[d] for m in range(M[i][j]) if i!=j) <= t[k,i])
                model.add_constraint(t[k,i] <= model.sum(x_m[i,j,k,m]*leave_time_end[i][j][m] for j in dsp_d_nodes[d] for m in range(M[i][j]) if i!=j)) 
    
    # Select only one time period to leave
    for d in D:
        for (i,j) in dsp_d_arcs[d]: 
            for k in vehicle_bounds_DSPs[d]:
                model.add_constraint(model.sum(x_m[i,j,k,m] for m in range(M[i][j])) == x[k,i,j])
    
    return model

def first_mile_follower(lamda, Pf, V1f, A1f, fol_depot, cost_per_km, num_vehicles_for_follower, locker_nodes, 
                        distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest):
    model = Model(name = 'First_Mile_Follower')
#     model.parameters.timelimit = 30
    
    # ----- Sets -----
    K = range(num_vehicles_for_follower)
    satellites = locker_nodes

    # ----- Variables ----- 
    # w_kij = 1 if arc (i,j) is traversed by vehicle k  
    w = {(k,i,j):model.binary_var(name='w_%d_%d_%d' % (k,i,j)) for k in K for i in V1f for j in V1f if i!=j}
        
    # Arrival time of vehicle k at node i - tau_ki
    tau = {(k,i):model.continuous_var(lb=0.0, name='tau_%d_%d' % (k,i)) for k in K for i in V1f}          
    
    
    # ----- Objective function: Minimize the the travel cost ----- 
    model.minimize(model.sum(cost_per_km*distance_matrix[i][j][0] * w[k,i,j] for k in K for (i,j) in A1f)
                  ) 
    
    # ----- Constraints -----             
    # Link assignment and routing variables
    for p in Pf:
        for s in satellites:
            model.add_constraint(model.sum(w[k,i,s] for i in V1f for k in K if i!=s) >= lamda[p,s])
    
    # Leave depot
    for k in K:
        model.add_constraint(model.sum(w[k,fol_depot,s] for s in satellites) == 1)
        
    # Flow conservation
    for i in V1f:
        for k in K:
            model.add_constraint(model.sum(w[k,j,i] for j in V1f if i!=j) - model.sum(w[k,i,j] for j in V1f if i!=j) == 0)  
        
    # Arrival time
    for (i,j) in A1f:
        if j==fol_depot: continue
        for k in K:
            model.add_constraint(tau[k,i] + travel_time_matrix[i][j][0] <= 
                                 tau[k,j] + (1 - w[k,i,j]) * bigM_matrix[i][j])
        
    # Earliest and latest time windows
    for k in K:
        for i in V1f:
            model.add_constraint(earliest[i]*model.sum(w[k,i,j] for j in V1f if i!=j) <= tau[k,i])
            model.add_constraint(tau[k,i] <= latest[i]*model.sum(w[k,i,j] for j in V1f if i!=j))
        
    # Solve
    solution = model.solve(log_output = False)   
    obj_val = None
    sol_df = None
    
    if model.solve_status.name == 'OPTIMAL_SOLUTION' or model.solve_status.name == 'FEASIBLE_SOLUTION':
        obj_val = solution.get_objective_value()
        sol_df = solution.as_df()
        
        
    return obj_val, sol_df

def last_mile_follower(y, lamda, V2d, A2d, fol_depot, cost_per_km, num_vehicles_for_follower, packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, num_time_periods_matrix, earliest, latest,
                       leave_time_start, leave_time_end, bigM_matrix, time_violation_penalty):
    model = Model(name = 'Last_Mile_Follower')
    model.parameters.timelimit = 250
    
    # --- Sets ---
    M = num_time_periods_matrix  
    K = range(num_vehicles_for_follower)
    P = packages 
    satellites = locker_nodes

    # ----- Variables ----- 
    # mu_kp = 1 if package p is caried by vehicle k
    mu = {(k,p):model.binary_var(name='mu_%d_%d' % (k,p)) for k in K for p in P}
        
    # x_kij = 1 if arc (i,j) is traversed by vehicle k  
    x = {(k,i,j):model.binary_var(name='x_%d_%d_%d' % (k,i,j)) for k in K for i in V2d for j in V2d if i!=j}
    
    # x_kmij = 1 if arc (i,j) is traversed by vehicle k in time period m     
    x_m = {(i,j,k,m):model.binary_var(name='xm_%d_%d_%d_%d' % (i,j,k,m)) for i in V2d
                                     for j in V2d if j!=i
                                     for k in K
                                     for m in range(M[i][j]) }
    
    # Arrival time of vehicle k at node i - t_ki
    t = {(k,i):model.continuous_var(lb=0.0, name='t_%d_%d' % (k,i)) for k in K for i in V2d}  
        
    # Variables for earliest and latest times
    alpha_early = {(k,i): model.continuous_var(lb=0.0, name='alphaEarly_%d_%d' % (k,i)) for k in K for i in V2d}
    alpha_late = {(k,i): model.continuous_var(lb=0.0, name='alphaLate_%d_%d' % (k,i)) for k in K for i in V2d}
    
    
    # ----- Objective function: Minimize the travel cost ----- 
    model.minimize(model.sum(cost_per_km*distance_matrix[i][j][m] * x_m[i,j,k,m] for (i,j) in A2d for k in K for m in range(M[i][j]))
                   + time_violation_penalty * model.sum(alpha_early[k,i] for k in K for i in V2d)
                   + time_violation_penalty * model.sum(alpha_late[k,i] for k in K for i in V2d)
                  )
    
    # ----- Time Violation Constraints ----- 
    for k in K:
        for i in V2d:
            model.add_constraint(alpha_early[k,i] >= earliest[i]*model.sum(x[k,i,j] for j in V2d if i!=j) - t[k,i])
            model.add_constraint(alpha_late[k,i] >= t[k,i] - latest[i]*model.sum(x[k,i,j] for j in V2d if i!=j))
    
    # ----- Constraints -----     
    
    # If a DSP is assigned a package, it should be carried on one vehicle k
    for p in P:
        model.add_constraint(model.sum(mu[k,p] for k in K) == y[p], "assign")
        
    # If a vehicle k picks up a package p, it should go to the destination of p
    for p in P:
        for k in K:
            model.add_constraint(model.sum(x[k,j,destinations[p]] for j in V2d if j != destinations[p]) == mu[k,p])
    
    # A package p should be picked up from its satellite s
    for p in P:
        for k in K:
            model.add_constraint(model.sum(lamda[p,s]*x[k,s,j] for s in satellites for j in V2d if s!=j) >= mu[k,p])        
            
    # A vehicle shoud go from a depot to a satellite
    for k in K:
        model.add_constraint(model.sum(x[k,fol_depot,s] for s in satellites) <= model.sum(mu[k,p] for p in P))
    
    # A vehicle may go from a satellite to another node
    for k in K:
        for s in satellites:
            model.add_constraint(model.sum(x[k,s,j] for j in V2d if s!=j) <= 1)
    
    # Flow conservation at all nodes
    for k in K:
        for j in V2d:    
            model.add_constraint(model.sum(x[k,i,j] for i in V2d if i!=j) - model.sum(x[k,j,i] for i in V2d if i!=j) == 0)
 
    
    # Satellite Time constraints
    for p in P:
        for k in K:
            model.add_constraint(model.sum(lamda[p,s]*(t[k,s] + min(travel_time_matrix[s][destinations[p]])) for s in satellites) <= 
                                 t[k, destinations[p]] + (1-mu[k,p])*100)#bigN[p][k])
    
    # Arrival time
    for (i,j) in A2d:
        if j==fol_depot: continue
        for k in K:
            model.add_constraint(t[k,i] + model.sum(x_m[i,j,k,m] * travel_time_matrix[i][j][m] for m in range(M[i][j]) if i!=j) <= 
                                 t[k,j] + (1 - x[k,i,j]) * bigM_matrix[i][j])

    # Time period
    for k in K:
        for i in V2d:
            model.add_constraint(model.sum(x_m[i,j,k,m]*leave_time_start[i][j][m] for j in V2d for m in range(M[i][j]) if i!=j) <= t[k,i])
            model.add_constraint(t[k,i] <= model.sum(x_m[i,j,k,m]*leave_time_end[i][j][m] for j in V2d for m in range(M[i][j]) if i!=j)) 
    
    # Select only one time period to leave
    for (i,j) in A2d: 
        for k in K:
            model.add_constraint(model.sum(x_m[i,j,k,m] for m in range(M[i][j])) == x[k,i,j])
    

  
    # Solve
    solution = model.solve(log_output = False)   
    obj_val = None
    sol_df = None

    
    if model.solve_status.name == 'OPTIMAL_SOLUTION' or model.solve_status.name == 'FEASIBLE_SOLUTION':
        obj_val = solution.get_objective_value()
        sol_df = solution.as_df()
        
        
    return obj_val, sol_df

def extract_alphas(hpr_sol, num_vehicles_per_DSP):
    alpha_early_df = None
    alpha_late_df = None
    
    if len(hpr_sol) > 0:
        alpha_early_rows = hpr_sol[hpr_sol['name'].str.startswith('alphaEarly')]
        alpha_early_rows = alpha_early_rows.reset_index(drop=True)  
        
        alpha_late_rows = hpr_sol[hpr_sol['name'].str.startswith('alphaLate')]
        alpha_late_rows = alpha_late_rows.reset_index(drop=True)
        
        vehicle_bounds = functions.create_bounds(num_vehicles_per_DSP)
        
        # Find out which vehicle belongs to which DSP
        list_of_vehicle_bounds = []
        for i in range(len(vehicle_bounds)):
            list_of_vehicle_bounds.append(list(vehicle_bounds[i]))
            
        dk_match = pd.DataFrame(columns = ['d', 'k'])
        for d in range(len(list_of_vehicle_bounds)):
            for k in list_of_vehicle_bounds[d]:
                dk_match.loc[len(dk_match)] = [d, float(k)]
                
               
        alpha_early_df = pd.DataFrame(columns=['k','i','value'])
        for i in range(alpha_early_rows.shape[0]):
            row = alpha_early_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(alpha_early_rows['value'][i])
            alpha_early_df.loc[len(alpha_early_df)] = row        
        alpha_early_df = alpha_early_df.merge(dk_match, on='k', how='left')
        
      
        alpha_late_df = pd.DataFrame(columns=['k','i','value'])
        for i in range(alpha_late_rows.shape[0]):
            row = alpha_late_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(alpha_late_rows['value'][i])
            alpha_late_df.loc[len(alpha_late_df)] = row
        alpha_late_df = alpha_late_df.merge(dk_match, on='k', how='left')
        
    return alpha_early_df,alpha_late_df 

def extract_xm(hpr_sol, num_vehicles_per_DSP, distance_matrix):
    xm_df = None
    
    if len(hpr_sol) > 0:
        xm_rows = hpr_sol[hpr_sol['name'].str.startswith('xm')]
        xm_rows = xm_rows.reset_index(drop=True)
        vehicle_bounds = functions.create_bounds(num_vehicles_per_DSP)

        # Remove possible duplicate
        if len(xm_rows) > 0:
            for i in range(len(xm_rows)):
                if xm_rows.loc[i]['value'] < 0.1:
                    xm_rows.drop([i], inplace=True)
        xm_rows.reset_index(inplace=True)

        # Find out which vehicle belongs to which DSP
        list_of_vehicle_bounds = []
        for i in range(len(vehicle_bounds)):
            list_of_vehicle_bounds.append(list(vehicle_bounds[i]))

        dk_match = pd.DataFrame(columns = ['d', 'k'])
        for d in range(len(list_of_vehicle_bounds)):
            for k in list_of_vehicle_bounds[d]:
                dk_match.loc[len(dk_match)] = [d, float(k)]

        xm_df = pd.DataFrame(columns=['i','j','k','m','c_ijm'])
        for i in range(xm_rows.shape[0]):
            row = xm_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(distance_matrix[row[0]][row[1]][row[3]])
            xm_df.loc[len(xm_df)] = row

        xm_df = xm_df.merge(dk_match, on='k', how='left')
        xm_df['d'] = xm_df['d'].astype('int')
    
      
    return xm_df

def extract_w(hpr_sol, num_vehicles_per_FM, distance_matrix):
    w_rows = hpr_sol[hpr_sol['name'].str.startswith('w')]
    w_rows = w_rows.reset_index(drop=True)
    vehicle_bounds = functions.create_bounds(num_vehicles_per_FM)
    
    # Remove possible duplicates
    if len(w_rows) > 0:
        for i in range(len(w_rows)):
            if w_rows.loc[i]['value'] < 0.1:
                w_rows.drop([i], inplace=True)
    w_rows.reset_index(inplace=True)
    
    # Find out which vehicle belongs to which First Miler
    list_of_vehicle_bounds = []
    for i in range(len(vehicle_bounds)):
        list_of_vehicle_bounds.append(list(vehicle_bounds[i]))
    
    fk_match = pd.DataFrame(columns = ['f', 'k'])
    
    for f in range(len(list_of_vehicle_bounds)):
        for k in list_of_vehicle_bounds[f]:
            fk_match.loc[len(fk_match)] = [f, float(k)]

    w_df = pd.DataFrame(columns=['k','i','j','c_ij'])
    for i in range(w_rows.shape[0]):
        row = w_rows['name'][i].split('_')
        row.pop(0);    
        row = [int(i) for i in row]
        row.append(distance_matrix[row[1]][row[2]][0])
        w_df.loc[len(w_df)] = row

    w_df = w_df.merge(fk_match, on='k', how='left')
    w_df['f'] = w_df['f'].astype('int')
    
    return w_df

# Compute FM followers'
def compute_first_mile_follower_obj(w_sol, num_FirstMilers, cost_per_km_for_FM):
    V_hat = []
    
    # For each FM follower...
    for f in range(num_FirstMilers):
        # If there is a solution...
        if len(w_sol) != 0:
            cost = w_sol.loc[w_sol['f'] == f, 'c_ij'].sum()
            cost = cost * cost_per_km_for_FM[f]
            V_hat.append(cost)
        else:
            V_hat.append(0)
        
    return V_hat

# Solve parameterized follower problem for each FM
def solve_param_first_mile_followers(lamda, Pf, num_FirstMilers, num_vehicles_per_FM, cost_per_km_for_FM, fm_depots, fm_f_nodes, fm_f_arcs,
                                     locker_nodes, distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest):
    V = []
    sol_df_vec = []
    
    for f in range(num_FirstMilers):
        obj, sol_df = first_mile_follower(lamda, Pf[f], fm_f_nodes[f], fm_f_arcs[f], fm_depots[f], cost_per_km_for_FM[f], num_vehicles_per_FM[f],
                                          locker_nodes, distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest) 
        

        

        V.append(obj) 
        sol_df_vec.append(sol_df)
    
    return V, sol_df_vec


# Compute DSP followers' response
def compute_last_mile_follower_obj(xm_sol, alpha_early_df, alpha_late_df, num_DSPs, cost_per_km_for_DSP):
    V_hat = []    
    total_cost_per_follower = []
    
    # For each follower...
    for d in range(num_DSPs):
        # If there is a solution...
        if len(xm_sol) != 0:
            cost = xm_sol.loc[xm_sol['d'] == d, 'c_ijm'].sum()
            cost = cost * cost_per_km_for_DSP[d] 
           
            penalty_early = alpha_early_df.loc[alpha_early_df['d'] == d, 'value'].sum()
            penalty_late= alpha_late_df.loc[alpha_late_df['d'] == d, 'value'].sum()
               
            total_cost_per_follower.append(cost + penalty_early + penalty_late)
        
            V_hat.append(cost)
        else:
            total_cost_per_follower.append(0)
            V_hat.append(0)
    
    return V_hat


# Solve parameterized follower problem for each DSP
def solve_param_last_mile_followers(y_sol, lamda_sol, num_DSPs, num_vehicles_per_DSP, cost_per_km_for_DSP, dsp_depots, dsp_d_nodes, dsp_d_arcs,
                                    packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, num_time_periods_matrix, earliest, latest,
                                    leave_time_start, leave_time_end, bigM_matrix, time_violation_penalty):
    V = []
    sol_df_vec = []

    for d in range(num_DSPs):  
        obj, sol_df = last_mile_follower(y_sol[:, d], lamda_sol, dsp_d_nodes[d], dsp_d_arcs[d], dsp_depots[d], cost_per_km_for_DSP[d], num_vehicles_per_DSP[d],
                                         packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, num_time_periods_matrix, earliest, latest, 
                                         leave_time_start, leave_time_end, bigM_matrix, time_violation_penalty) 
        
        
        
        V.append(obj)
        sol_df_vec.append(sol_df)
        
    return V, sol_df_vec

def do_check(epss, diff):    
    checks = []*len(diff)

    for item in diff:
        if item > epss:
            c = True
        else:
            c = False
        checks.append(c)

    return checks

def solve_HPR_model(model, packages, num_nodes, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, verboseTorF):    
    # Solve
    solution = model.solve(log_output = verboseTorF)
    # print('Status:', model.solve_status.name)
    
    
    hpr_obj_value = solution.get_objective_value()
    # print('Objective value:', hpr_obj_value)
    
    hpr_sol = None
    lamda_sol = None
    y_sol = None
    
    if model.solve_status.name == 'OPTIMAL_SOLUTION' or model.solve_status.name == 'FEASIBLE_SOLUTION':
        hpr_sol = solution.as_df()        
        
        w_sol = extract_w(hpr_sol, num_vehicles_per_FM, distance_matrix)      
        xm_sol = extract_xm(hpr_sol, num_vehicles_per_DSP, distance_matrix)
        alpha_early_sol, alpha_late_sol = extract_alphas(hpr_sol, num_vehicles_per_DSP)
             
        lamda_sol = np.zeros((len(packages), num_nodes))
        for p in range(len(packages)):
            for n in range(num_nodes):
                lamda_sol[p,n] = model.get_var_by_name('lamda_%d_%d' % (p,n))
        # removing numerical inconsistencies
        for p in range(len(packages)):
            for n in range(num_nodes):
                if lamda_sol[p,n] != np.nan:
                    if lamda_sol[p,n] <= 0.1:
                        lamda_sol[p,n] =  0.0
                            
        y_sol = np.zeros((len(packages), num_DSPs))
        for p in range(len(packages)):
            for d in range(num_DSPs):
                y_sol[p,d] = model.get_var_by_name('y_%d_%d' % (p,d))
        # removing numerical inconsistencies
        for p in range(len(packages)):
            for d in range(num_DSPs):
                if y_sol[p,d] <= 0.1:
                    y_sol[p,d] = 0.0               
            
    return hpr_sol, lamda_sol, w_sol, xm_sol, y_sol, hpr_obj_value, alpha_early_sol, alpha_late_sol


def cutting_plane_algorithm(hpr_mod, Pf, packages, destinations, locker_nodes, num_nodes, num_FirstMilers, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, cost_per_km_for_FM, cost_per_km_for_DSP, 
                            distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs, 
                            num_time_periods_matrix, leave_time_start, leave_time_end, time_violation_penalty, verboseTorF):
    num_iterations = 0
    epsilon = 1e-4
    bigMcutD = 1e6
    bigMcutF = 1e6
    convergedFirst = False
    convergedLast = False
    converged = False
    
    veh_bounds_FM = functions.create_bounds(num_vehicles_per_FM)
    veh_bounds_DSP = functions.create_bounds(num_vehicles_per_DSP)
   
    while converged == False:
        print('\n>>> Iteration:', num_iterations)
        
        # 1. Solve HPR to get lamda, w, x, y
        print('Solving HPR...')
        hpr_sol, lamda_sol, w_sol, xm_sol, y_sol, hpr_obj_value, alpha_early_sol, alpha_late_sol = solve_HPR_model(hpr_mod, packages, num_nodes, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, 
                                                                                  distance_matrix, verboseTorF)
        
                            
        # 2. For each follower:        
        #    - evaluate FM follower response
        V_hat_FirstM = compute_first_mile_follower_obj(w_sol, num_FirstMilers, cost_per_km_for_FM)
        # print('V_hat_FirstM = ', V_hat_FirstM)
        #    - solve FM Follower(lamda)
        print('Solving First-Mile follower problems...')
        V_FirstM, first_m_sol_dfs = solve_param_first_mile_followers(lamda_sol, Pf, num_FirstMilers, num_vehicles_per_FM, cost_per_km_for_FM, fm_depots, fm_f_nodes, fm_f_arcs,
                                                                     locker_nodes, distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest)
                
        # print('V_FirstM = ', V_FirstM)
        #    - First Mile: If  V_f_hat - V_f > epsilon, generate optimality cut
        difference_FM = list(np.array(V_hat_FirstM) - np.array(V_FirstM))
        
        #    - evaluate LM follower response to y
        V_hat_LastM = compute_last_mile_follower_obj(xm_sol, alpha_early_sol, alpha_late_sol, num_DSPs, cost_per_km_for_DSP)
        # print('V_hat_LastM = ', V_hat_LastM)
        #    - solve LM Follower(y)
        print('Solving Last-Mile follower problems...')
        V_LastM, last_m_sol_dfs = solve_param_last_mile_followers(y_sol, lamda_sol, num_DSPs, num_vehicles_per_DSP, cost_per_km_for_DSP, dsp_depots, dsp_d_nodes, dsp_d_arcs,
                                                                  packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, num_time_periods_matrix, earliest, latest,
                                                                  leave_time_start, leave_time_end, bigM_matrix, time_violation_penalty)
        
        
               
        
        
        # print('V_LastM = ', V_LastM)                
        #    - Last Mile: If V_d_hat - V_d > epsilon, generate optimality cut
        difference_LM = list(np.array(V_hat_LastM) - np.array(V_LastM))     

        
        checks_FM = do_check(epsilon, difference_FM)
        # print('checks_FM:', checks_FM)
        
        checks_LM = do_check(epsilon, difference_LM)
        # print('checks_LM:', checks_LM)
        
        if any(checks_FM): 
            # print('First Mile Gap:', difference_FM)
            print('Adding FM cuts...')
            # Add optimality cuts to HPR  for violating follower only
            for f in range(num_FirstMilers):
                if checks_FM[f] == True:
                    
                    
                    # get indices where lamda_p_s == 1
                    one_indices = np.argwhere(lamda_sol[Pf[f], :] == 1)
                    # Split one_indices into two
                    one_p_indices = one_indices[:, 0]
                    one_s_indices = one_indices[:, 1]

                    # get indices where lamda_p_s == 0
                    zero_indices = np.argwhere(lamda_sol[Pf[f], :] == 0)
                    # Split zero_indices into two
                    zero_p_indices = zero_indices[:, 0]
                    zero_s_indices = zero_indices[:, 1]

                    value_ones = hpr_mod.sum(lamda_sol[Pf[f], :][p][s] - hpr_mod.get_var_by_name('lamda_%d_%d' % (p,s)) for p in one_p_indices for s in one_s_indices)
                    value_zeros = hpr_mod.sum(hpr_mod.get_var_by_name('lamda_%d_%d' % (p,s)) - lamda_sol[Pf[f], :][p][s] for p in zero_p_indices for s in zero_s_indices)

                    # add constraint
                    hpr_mod.add_constraint(hpr_mod.sum(cost_per_km_for_FM[f]*distance_matrix[i][j][0] * hpr_mod.get_var_by_name('w_%d_%d_%d' % (k,i,j)) for k in veh_bounds_FM[f] for i in fm_f_nodes[f] for j in fm_f_nodes[f] if i!=j)   
                                               <= V_FirstM[f] + (value_ones + value_zeros)*bigMcutF)
        else:
            convergedFirst = True


        if any(checks_LM): 
            # print('Last Mile Gap:', difference_LM)
            print('Adding LM cuts...')
            # Add optimality cuts to HPR  for violating follower only
            for d in range(num_DSPs):
                if checks_LM[d] == True:
                    # get rows where y_sol == 1
                    one_indices = np.argwhere(y_sol[:, d] == 1).flatten() 
                    # get rows where y_sol == 0
                    zero_indices = np.argwhere(y_sol[:, d] == 0).flatten()                

                    value_ones = hpr_mod.sum(y_sol[:, d][p] - hpr_mod.get_var_by_name('y_%d_%d' % (p,d)) for p in one_indices)
                    value_zeros = hpr_mod.sum(hpr_mod.get_var_by_name('y_%d_%d' % (p,d)) - y_sol[:, d][p] for p in zero_indices)

                    # add constraint
                    hpr_mod.add_constraint(hpr_mod.sum(cost_per_km_for_DSP[d]*distance_matrix[i][j][m] * hpr_mod.get_var_by_name('xm_%d_%d_%d_%d' % (i,j,k,m)) for i in dsp_d_nodes[d] for j in dsp_d_nodes[d] for k in veh_bounds_DSP[d] for m in range(num_time_periods_matrix[i][j]) if i!=j)
                                           + time_violation_penalty * hpr_mod.sum(hpr_mod.get_var_by_name('alphaEarly_%d_%d' % (k,i)) for k in veh_bounds_DSP[d] for i in dsp_d_nodes[d])
                                           + time_violation_penalty * hpr_mod.sum(hpr_mod.get_var_by_name('alphaLate_%d_%d' % (k,i)) for k in veh_bounds_DSP[d] for i in dsp_d_nodes[d])
                                           <= V_LastM[d] + (value_ones + value_zeros)*bigMcutD)
        else:
            convergedLast = True

        # update number of iterations
        num_iterations += 1   
        
        if convergedFirst == True and convergedLast == True:
            # All followers are optimal and we can stop
            print('All followers optimal. Terminating.')
            # print('FM Gap:', difference_FM)
            # print('LM Gap:', difference_LM)
            converged = True  
    
    return hpr_sol, lamda_sol, w_sol, xm_sol, y_sol