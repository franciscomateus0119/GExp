import docplex
import pandas as pd
import tensorflow as tf
import numpy as np
import utility
import copy
from milp import codify_network
from teste import get_minimal_explanation

def model_classification_output(k_model, net_input):
    #Prediction for a single sample
    net_input = tf.reshape(tf.constant(net_input), [1, -1])
    net_output = k_model.predict(tf.constant(net_input), verbose = 0)[0]
    net_output = tf.argmax(net_output)

    predictions = k_model.predict(tf.constant(net_input), verbose = 0)[0, 0]

    #print(f'Predictions: (ndarray[ndarray[{type(predictions)}]])', predictions)
    classification: np.int64 = net_output.numpy()
    #print(f'Network output: ({type(classification)})', classification)
    return net_input, net_output

def find_minimal(model, net_input, net_output_, number_of_classes, output_bnds, method='fischetti'):
    return model, get_minimal_explanation(model, net_input, net_output_, number_of_classes, method, output_bnds)

def remove_equality_constraints(model, return_linear_constraints = True):
    for constraint in model.find_matching_linear_constraints('input'):
        model.remove_constraint(constraint)
        model.add_constraint(constraint.lhs <= constraint.rhs.clone(), 'input LE')
        model.add_constraint(constraint.lhs >= constraint.rhs.clone(), 'input GE')
    if return_linear_constraints:
        return model, model.find_matching_linear_constraints('input')
    return model

def find_bounds(minimal_explanation: list):
    for constraint in minimal_explanation:
        #testing_model.solve()
        print('Initial constraint:' + '\t', constraint)

        variable = constraint.lhs
        print(f"variable {variable} ub {variable.ub}")
        print(f"variable {variable} lb {variable.lb}")

def find_constraints(minimal_explanation: list):
    for constraint in minimal_explanation:
        print(constraint)
        if constraint.sense == docplex.mp.constants.ComparisonType.GE:
            print('\n')
            
def find_ranges_no_log(model, minimal_explanation: list, epsilon: float):
    constraint_LE = None
    constraint_GE = None
    ranges = []
    for i, constraint in enumerate(minimal_explanation):
        variable = constraint.lhs
        constraint_val = constraint.rhs.constant
        #Check if its a (var <= value) constraint
        if constraint.sense == docplex.mp.constants.ComparisonType.LE: 
            #the value in(var <= value) is set to the maximum possible, i.e. the upper bound
            constraint.rhs = variable.ub            
            model.minimize(variable)            
            sol = model.solve()
            if sol:
                if constraint_val <= (model.objective_value - epsilon):
                    constraint_LE = model.objective_value - epsilon                 
                else:
                    constraint_LE = constraint_val                 
            else:
                constraint_LE = variable.ub
            constraint.rhs = constraint_LE

        #Check if its a (var >= value)
        elif constraint.sense == docplex.mp.constants.ComparisonType.GE:
            constraint.rhs = variable.lb            
            model.maximize(variable)            
            sol = model.solve()
            if sol:
                if constraint_val >= (model.objective_value + epsilon):
                    constraint_GE = model.objective_value + epsilon
                else:
                    constraint_GE = constraint_val
            else:
                constraint_GE = variable.lb

            constraint.rhs = constraint_GE
        else:
            raise Exception('Constraint sense was neither LE nor GE')
        if (constraint_LE is not None and constraint_GE is not None):
            ranges.append([constraint_GE, constraint_LE])
            constraint_LE = None
            constraint_GE = None 
    return ranges

def find_ranges(model, minimal_explanation: list, epsilon: float):
    constraint_LE = None
    constraint_GE = None
    ranges = []
    for i, constraint in enumerate(minimal_explanation):
        print('Initial constraint:' + '\t', constraint)
        print(f"Current GE and LE: {constraint_GE}, {constraint_LE}")
        variable = constraint.lhs
        constraint_val = constraint.rhs.constant
        #Check if its a (var <= value) constraint
        if constraint.sense == docplex.mp.constants.ComparisonType.LE: 
            #the value in(var <= value) is set to the maximum possible, i.e. the upper bound
            constraint.rhs = variable.ub
            print(f"LE constraint set to upper bound {variable.ub} --> {constraint}")
            
            model.minimize(variable)            
            sol = model.solve()
            if sol:
                print(f"Changed class with value = {model.objective_value}")
                print(f"Diference between original value and solver found value: {abs(constraint_val) - abs(model.objective_value)}")
                print(f"Epsilon >= Diference? {abs(abs(constraint_val) - abs(model.objective_value)) <= epsilon}")
                if constraint_val <= (model.objective_value - epsilon):
                    print(constraint_val, (model.objective_value - epsilon))
                    constraint_LE = model.objective_value - epsilon
                    
                else:
                    print("[MIN] Reducing epsilon overpasses original value.")
                    constraint_LE = constraint_val
                    
            else:
                print(f"Class not changed")
                constraint_LE = variable.ub
            #constraint.rhs = constraint_val
            #print(f"Reseted constraint to {constraint_val} --> {constraint} ")
            constraint.rhs = constraint_LE

        #Check if its a (var >= value)
        elif constraint.sense == docplex.mp.constants.ComparisonType.GE:
            constraint.rhs = variable.lb
            print(f"GE constraint set to lower bound {variable.lb} --> {constraint}")
            
            model.maximize(variable)            
            sol = model.solve()
            if sol:
                print(f"Changed class with value = {model.objective_value}")
                print(f"Diference between original value and solver found value: {abs(constraint_val) - abs(model.objective_value)}")
                print(f"Epsilon >= Diference? {abs(abs(constraint_val) - abs(model.objective_value)) <= epsilon}")
                if constraint_val >= (model.objective_value + epsilon):
                    print(f"constraint_val = {constraint_val}, new_value = {model.objective_value + epsilon}")
                    constraint_GE = model.objective_value + epsilon
                else:
                    print("[MAX] Reducing epsilon overpasses original value.")
                    constraint_GE = constraint_val
            else:
                print(f"Class not changed")
                constraint_GE = variable.lb

            constraint.rhs = constraint_GE          
        else:
            raise Exception('Constraint sense was neither LE nor GE')
        if (constraint_LE is not None and constraint_GE is not None):
            print(f"Found bounds for {variable}: {constraint_GE}, {constraint_LE}")
            ranges.append([constraint_GE, constraint_LE])
            constraint_LE = None
            constraint_GE = None 
        print("\n")
    return(ranges)

def beautify_output(model, og_bounds, found_ranges, minimal_exp):
    ranges = copy.deepcopy(og_bounds)
    found_bounds = model.find_matching_linear_constraints('input')
    indexes = []
    for min_exp in minimal_exp:
        indexes.append(int(min_exp.lhs.name.replace("x_","")))
    #print(f"Indexes = {indexes}")
    #print(f"Original ranges {ranges}")
    for idx, val in zip(indexes, found_ranges):
        ranges[idx][0] = val[0]
        ranges[idx][1] = val[1]
        #print(f"Index {idx}, val {val}, ranges {ranges}")
        
    return ranges

def run_explanation(sample, n_classes, kmodel, model, output_bounds, og_bounds, epsilon = 0.0001, enable_log = False, enable_display_coverage = False):
    network_input, network_output = model_classification_output(k_model=kmodel, net_input=sample)
    result_model, minimal_explanation = find_minimal(model.clone(), network_input, network_output, n_classes, output_bounds)
    result_model, linear_constraints = remove_equality_constraints(result_model.clone())
    if enable_log:
        found_ranges = find_ranges(result_model, linear_constraints, epsilon = epsilon)
        print(f"Found ranges {found_ranges}")
        #print(find_ranges(result_model, linear_constraints, epsilon = epsilon))
        explanation = beautify_output(result_model,og_bounds, found_ranges, minimal_explanation)
        return explanation, minimal_explanation
    found_ranges = find_ranges_no_log(result_model, linear_constraints, epsilon = epsilon)
    explanation = beautify_output(result_model,og_bounds, found_ranges, minimal_explanation)
    return explanation, minimal_explanation

def find_ranges_doublestep(model, minimal_explanation: list, epsilon: float,  p = 0.25):
    #Step 1 - Small Opening
    constraint_LE = None
    constraint_GE = None
    for i, constraint in enumerate(minimal_explanation):
        print('Initial constraint:' + '\t', constraint)
        variable = constraint.lhs
        constraint_val = constraint.rhs.constant
        #Check if its a (var <= value) constraint
        if constraint.sense == docplex.mp.constants.ComparisonType.LE: 
            #the value in(var <= value) is set to the maximum possible, i.e. the upper bound
            constraint.rhs = variable.ub
            print(f"LE constraint set to upper bound {variable.ub} --> {constraint}")
            model.minimize(variable)            
            sol = model.solve()
            if sol:
                print(f"Changed class with value = {model.objective_value}")
                #relevance_value[1] + (X_minimum[exclude].solution_value  - precision - relevance_value[1])*p
                new_bound_value = constraint_val + (model.objective_value - epsilon - constraint_val) * p
                print(f"Value with p = {p} --> {new_bound_value}")
                print(f"Diference between original value and solver found value: {abs(constraint_val) - abs(model.objective_value)}")
                print(f"Diference between original value and new_value: {abs(constraint_val) - abs(new_bound_value)}")
                print(f"Epsilon >= Diference (new_value)? {abs(abs(constraint_val) - abs(new_bound_value)) <= epsilon}")
                if constraint_val <= (new_bound_value ):
                    print(constraint_val, (new_bound_value))
                    constraint_LE = new_bound_value
                    
                else:
                    print("[MIN] Reducing epsilon overpasses original value.")
                    
                    constraint_LE = constraint_val
                    
            else:
                print(f"Class not changed")
                #relevance_value[1] + (upper_bound - relevance_value[1])*p
                constraint_LE =  constraint_val + (variable.ub -  + (constraint_val)) * p
            #constraint.rhs = constraint_val
            #print(f"Reseted constraint to {constraint_val} --> {constraint} ")
            constraint.rhs = constraint_LE

        #Check if its a (var >= value)
        elif constraint.sense == docplex.mp.constants.ComparisonType.GE:
            constraint.rhs = variable.lb
            print(f"GE constraint set to lower bound {variable.lb} --> {constraint}")
            model.maximize(variable)            
            sol = model.solve()
            if sol:
                #relevance_value[0] = relevance_value[0] + (X_maximum[exclude].solution_value + precision - relevance_value[0])*p
                new_bound_value = constraint_val + (model.objective_value + epsilon - constraint_val) * p
                print(f"Value with p = {p} --> {new_bound_value}")
                print(f"Diference between original value and solver found value: {abs(constraint_val) - abs(model.objective_value)}")
                print(f"Diference between original value and new_value: {abs(constraint_val) - abs(new_bound_value)}")
                print(f"Epsilon >= Diference (new_value)? {abs(abs(constraint_val) - abs(new_bound_value)) <= epsilon}")
                if constraint_val >= new_bound_value:
                    constraint_GE = new_bound_value
                else:
                    print("[MAX] Reducing epsilon overpasses original value.")
                    constraint_GE = constraint_val
            else:
                print(f"Class not changed")
                #relevance_value[0] = relevance_value[0] + (lower_bound - relevance_value[0])*p
                constraint_GE = constraint_val + (variable.lb - constraint_val) * p

            constraint.rhs = constraint_GE
        else:
            raise Exception('Constraint sense was neither LE nor GE')
        if (constraint_LE is not None and constraint_GE is not None):
                print(f"Found bounds for {variable}: {constraint_GE}, {constraint_LE}")
                constraint_LE = None
                constraint_GE = None
        print("\n")
    print("END OF STEP 1\n")
    print("START OF STEP 2\n")
    #Step 2 - Full Opening
    constraint_LE = None
    constraint_GE = None
    ranges = []
    for i, constraint in enumerate(minimal_explanation):
        print('Initial constraint:' + '\t', constraint)
        print(f"Step2 Constriant_LE {constraint_LE}, Constraint_GE {constraint_GE}")
        variable = constraint.lhs
        constraint_val = constraint.rhs.constant
        #Check if its a (var <= value) constraint
        if constraint.sense == docplex.mp.constants.ComparisonType.LE: 
            #the value in(var <= value) is set to the maximum possible, i.e. the upper bound
            constraint.rhs = variable.ub
            print(f"LE constraint set to upper bound {variable.ub} --> {constraint}")
            
            model.minimize(variable)            
            sol = model.solve()
            if sol:
                print(f"Changed class with value = {model.objective_value}")
                print(f"Diference between original value and solver found value: {abs(constraint_val) - abs(model.objective_value)}")
                print(f"Epsilon >= Diference? {abs(abs(constraint_val) - abs(model.objective_value)) <= epsilon}")
                if constraint_val <= (model.objective_value - epsilon):
                    print(constraint_val, (model.objective_value - epsilon))
                    constraint_LE = model.objective_value - epsilon
                    
                else:
                    print("[MIN] Reducing epsilon overpasses original value.")
                    constraint_LE = constraint_val
                    
            else:
                print(f"Class not changed")
                constraint_LE = variable.ub
            #constraint.rhs = constraint_val
            #print(f"Reseted constraint to {constraint_val} --> {constraint} ")
            constraint.rhs = constraint_LE

        #Check if its a (var >= value)
        elif constraint.sense == docplex.mp.constants.ComparisonType.GE:
            constraint.rhs = variable.lb
            print(f"GE constraint set to lower bound {variable.lb} --> {constraint}")
            
            model.maximize(variable)            
            sol = model.solve()
            if sol:
                print(f"Changed class with value = {model.objective_value}")
                print(f"Diference between original value and solver found value: {abs(constraint_val) - abs(model.objective_value)}")
                print(f"Epsilon >= Diference? {abs(abs(constraint_val) - abs(model.objective_value)) <= epsilon}")
                if constraint_val >= (model.objective_value + epsilon):
                    constraint_GE = model.objective_value + epsilon
                else:
                    print("[MAX] Reducing epsilon overpasses original value.")
                    constraint_GE = constraint_val
            else:
                print(f"Class not changed")
                constraint_GE = variable.lb

            constraint.rhs = constraint_GE
              
        else:
            raise Exception('Constraint sense was neither LE nor GE')
        if (constraint_LE is not None and constraint_GE is not None):
            print(f"Found bounds for {variable}: {constraint_GE}, {constraint_LE}")
            ranges.append([constraint_GE, constraint_LE])
            print(f'RANGES = {ranges}')
            constraint_LE = None
            constraint_GE = None
        print("\n")
    return ranges

def find_ranges_doublestep_no_log(model, minimal_explanation: list, epsilon: float,  p = 0.25):
    #Step 1 - Small Opening
    constraint_LE = None
    constraint_GE = None
    for i, constraint in enumerate(minimal_explanation):
        variable = constraint.lhs
        constraint_val = constraint.rhs.constant
        #Check if its a (var <= value) constraint
        if constraint.sense == docplex.mp.constants.ComparisonType.LE: 
            #the value in(var <= value) is set to the maximum possible, i.e. the upper bound
            constraint.rhs = variable.ub
            model.minimize(variable)            
            sol = model.solve()
            if sol:
                new_bound_value = constraint_val + (model.objective_value - epsilon - constraint_val) * p
                if constraint_val <= (new_bound_value ):
                    constraint_LE = new_bound_value     
                else:                    
                    constraint_LE = constraint_val        
            else:
                constraint_LE =  constraint_val + (variable.ub -  + (constraint_val)) * p
            constraint.rhs = constraint_LE

        #Check if its a (var >= value)
        elif constraint.sense == docplex.mp.constants.ComparisonType.GE:
            constraint.rhs = variable.lb
            model.maximize(variable)            
            sol = model.solve()
            if sol:
                new_bound_value = constraint_val + (model.objective_value + epsilon - constraint_val) * p
                if constraint_val >= new_bound_value:
                    constraint_GE = new_bound_value
                else:
                    constraint_GE = constraint_val
            else:
                constraint_GE = constraint_val + (variable.lb - constraint_val) * p

            constraint.rhs = constraint_GE
            
        else:
            raise Exception('Constraint sense was neither LE nor GE')
        if (constraint_LE is not None and constraint_GE is not None):
            constraint_LE = None
            constraint_GE = None
            
    #Step 2 - Full Opening
    constraint_LE = None
    constraint_GE = None
    ranges = []
    for i, constraint in enumerate(minimal_explanation):
        variable = constraint.lhs
        constraint_val = constraint.rhs.constant
        #Check if its a (var <= value) constraint
        if constraint.sense == docplex.mp.constants.ComparisonType.LE: 
            #the value in(var <= value) is set to the maximum possible, i.e. the upper bound
            constraint.rhs = variable.ub            
            model.minimize(variable)            
            sol = model.solve()
            if sol:
                if constraint_val <= (model.objective_value - epsilon):
                    constraint_LE = model.objective_value - epsilon
                else:
                    constraint_LE = constraint_val         
            else:
                constraint_LE = variable.ub
            constraint.rhs = constraint_LE

        #Check if its a (var >= value)
        elif constraint.sense == docplex.mp.constants.ComparisonType.GE:
            constraint.rhs = variable.lb            
            model.maximize(variable)            
            sol = model.solve()
            if sol:
                if constraint_val >= (model.objective_value + epsilon):
                    constraint_GE = model.objective_value + epsilon
                else:
                    constraint_GE = constraint_val
            else:
                constraint_GE = variable.lb
            constraint.rhs = constraint_GE
            
        else:
            raise Exception('Constraint sense was neither LE nor GE')
        if (constraint_LE is not None and constraint_GE is not None):
            ranges.append([constraint_GE, constraint_LE])
            constraint_LE = None
            constraint_GE = None
    return ranges

def run_explanation_doublestep(sample, n_classes, kmodel, model, output_bounds, og_bounds, epsilon = 0.0001, p = 0.5, enable_log = False, enable_display_coverage = False):
    network_input, network_output = model_classification_output(k_model=kmodel, net_input=sample)
    result_model, minimal_explanation = find_minimal(model.clone(), network_input, network_output, n_classes, output_bounds)
    result_model, linear_constraints = remove_equality_constraints(result_model.clone())
    if enable_log:
        found_ranges = find_ranges_doublestep(result_model, linear_constraints, epsilon = epsilon, p=p)
        print(f"Found ranges {found_ranges}")
        #print(find_ranges(result_model, linear_constraints, epsilon = epsilon))
        explanation = beautify_output(result_model,og_bounds, found_ranges, minimal_explanation)
        return explanation, minimal_explanation
    found_ranges = find_ranges_doublestep_no_log(result_model, linear_constraints, epsilon = epsilon, p=p)
    explanation = beautify_output(result_model,og_bounds, found_ranges, minimal_explanation)
    return explanation, minimal_explanation