# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:23:06 2024

@author: ade.fajemisin
"""
from docplex.mp.model import Model
import functions
import pandas as pd

def Last_Mile_Follower(y, lamda, V2d, A2d, fol_depot, locker_nodes, packages, destinations, num_vehicles_for_follower,
                      distance_matrix, travel_time_matrix, num_time_periods_matrix, bigM_matrix,
                    earliest, latest, leave_time_start, leave_time_end, emissions_matrix_EV, time_violation_penalty, time_limit_seconds):
    model = Model(name = 'Last_Mile_Follower')
    model.parameters.timelimit = time_limit_seconds
    
    # --- Sets ---
    
    P = packages    
    satellites = locker_nodes
        
    M = num_time_periods_matrix  
    K = range(num_vehicles_for_follower)

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
    
    
    # ----- Objective function: Minimize the emissions travel cost ----- 
    model.minimize(model.sum(emissions_matrix_EV[i][j][m] * x_m[i,j,k,m] for (i,j) in A2d for k in K for m in range(M[i][j]))
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

def solve_param_last_mile_followers(y, lamda, dsp_d_nodes, dsp_d_arcs,
                                    dsp_depots, locker_nodes, packages, destinations, 
                                    num_vehicles_per_DSP, distance_matrix, travel_time_matrix, 
                                    num_time_periods_matrix, bigM_matrix, earliest, latest, 
                                    leave_time_start, leave_time_end, emissions_matrix_EV, time_violation_penalty, 
                                    time_limit_seconds):
    V = []
    sol_df_vec = []
    
    num_DSPs = len(dsp_d_nodes)    

    for d in range(num_DSPs):  
        print('Solving for DSP', d)
        print('Time limit for  DSP ', d, 'is ', time_limit_seconds, 'seconds')
        obj, sol_df = Last_Mile_Follower(y[:, d], lamda, dsp_d_nodes[d], dsp_d_arcs[d], dsp_depots[d], locker_nodes, packages, destinations, num_vehicles_per_DSP[d],
                      distance_matrix, travel_time_matrix, num_time_periods_matrix, bigM_matrix,
                    earliest, latest, leave_time_start, leave_time_end, emissions_matrix_EV, time_violation_penalty, time_limit_seconds)
    
        V.append(obj)
        sol_df_vec.append(sol_df)
        
    return V, sol_df_vec

def extract_xm(sol_df, num_vehicles_per_DSP, distance_matrix):
    xm_df = None
    
    if len(sol_df) > 0:
        xm_rows = sol_df[sol_df['name'].str.startswith('xm')]
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

def extract_t(lastmiler_final_sol):
    t_df = None
    
    if len(lastmiler_final_sol) > 0:
        t_rows = lastmiler_final_sol[lastmiler_final_sol['name'].str.startswith('t_')]
        t_rows = t_rows.reset_index(drop=True)

        # Remove possible duplicate
        if len(t_rows) > 0:
            for i in range(len(t_rows)):
                if t_rows.loc[i]['value'] < 0.01:
                    t_rows.drop([i], inplace=True)
        t_rows.reset_index(inplace=True)

        t_df = pd.DataFrame(columns=['k','i','time'])
        for i in range(t_rows.shape[0]):
            row = t_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(t_rows['value'][i])
            t_df.loc[len(t_df)] = row    
      
    return t_df