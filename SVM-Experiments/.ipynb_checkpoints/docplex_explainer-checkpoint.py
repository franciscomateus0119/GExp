import docplex.mp.model
import docplex
import numpy as np

def twostep(classifier, dual_coef, support_vectors, intercept, data, show_log = 0, lower_bound = 0, upper_bound = 1,
                                       precision = 0.01,noise = 0.0001, threshold = 0, positive=True, p = 0.1, calculate_diag = False,  problem_name = "Twostep_Explanation"):
    
    if type(data) != np.ndarray:
        try:
            data = np.asarray(data)
        except exception:
            print(f"Data type {type(data)} could not be changed to type numpy.ndarray")   
    if data.ndim >= 2 and len(data) > 1:
        raise UserWarning("More than one pattern has been received. Must pass one pattern only") 
    if p < 0 or p > 1:
        raise UserWarning("Value for p is invalid. Must be within 0 < p <= 1")
    # Get minimal explanation    
    minimal = minimal_explanation(
            classifier = classifier,
            dual_coef = dual_coef,
            support_vectors = support_vectors,
            intercept = intercept,
            lower_bound = lower_bound,
            upper_bound = upper_bound,
            data = np.atleast_2d(data),
            positive = positive)[0]
    #print(f'Minimal: {minimal}')
    data = np.ravel(data)
    
    # Create optimization problem/variables
    model_min = docplex.mp.model.Model()
    model_max = docplex.mp.model.Model()
    explanations = []
    X_minimum = np.asarray([model_min.continuous_var(lower_bound, upper_bound, 'x'+str(i)) for i in range(len(data))]) # Optimization Variables for Minimize model
    X_maximum = np.asarray([model_max.continuous_var(lower_bound, upper_bound, 'x_'+str(i)) for i in range(len(data))]) # Optimization Variables for Maximize model
    
    # R restriction according to c class (Positive or Negative)
    if positive:
        model_min.add_constraint(((dual_coef @ support_vectors) @ X_minimum.reshape(1, len(X_minimum)).T + intercept)[0][0] <= threshold -noise)
        model_max.add_constraint(((dual_coef @ support_vectors) @ X_maximum.reshape(1, len(X_maximum)).T + intercept)[0][0] <= threshold -noise)

    else:
        model_min.add_constraint(((dual_coef @ support_vectors) @ X_minimum.reshape(1, len(X_minimum)).T + intercept)[0][0] >= threshold +noise)
        model_max.add_constraint(((dual_coef @ support_vectors) @ X_maximum.reshape(1, len(X_maximum)).T + intercept)[0][0] >= threshold +noise)
        
    # Get ranges from minimal explanation
    ranges = [] # All feature ranges
    ranges_minimal_index = [] # Store the indexes of relevant features
    for i in range(len(X_minimum)):
        if i in minimal:
            ranges.append([data[i],data[i]])
            ranges_minimal_index.append(i)
            X_minimum[i].lb = data[i]
            X_minimum[i].ub = data[i]
            X_maximum[i].lb = data[i]
            X_maximum[i].ub = data[i]
        else:
            ranges.append([lower_bound, upper_bound])
            X_minimum[i].lb = lower_bound
            X_minimum[i].ub = upper_bound
            X_maximum[i].lb = lower_bound
            X_maximum[i].ub = upper_bound
            
    # Auxiliary Variables 
    checked = [] # Stores checked features
    features_ranges = [] # Stores found explanation ranges
    not_relevant = [] #Stores irrelevant features

    #For every feature    
    for j in (minimal):
        
        #The feature to be checked
        exclude = j
        
        # Set Optimization bounds for the Minimize model
        X_minimum[exclude].lb = data[exclude] # Original Value
        X_minimum[exclude].ub = upper_bound # Upperbound
        model_min.minimize(X_minimum[exclude]) # Cost Function
        
        # Set Optimization bounds for the Maximize model
        X_maximum[exclude].lb = lower_bound # Lowerbound
        X_maximum[exclude].ub = data[exclude] # Original Value
        model_max.maximize(X_maximum[exclude]) # Cost Function
        
        # Store feature bounds. Starts with the original value of the feature.
        feature_bounds = [data[exclude], data[exclude]]
        
        # Solve Maximization problem to find the new Lowerbound.
        sat_maximum = model_max.solve(log_output=False)
        
        # If the Maximization problem is satisfiable.
        if sat_maximum:         
            # Update Lowerbound
            feature_bounds[0] = feature_bounds[0] + (X_maximum[exclude].solution_value + precision - feature_bounds[0])*p
            
            # If the updated Lowerbound value is higher than the original feature value, undo update.
            if feature_bounds[0] >= data[exclude]:
                feature_bounds[0] = data[exclude]
        
        # If the problem is unsatisfiable
        else:
            # Lowerbound unchaged
            feature_bounds[0] = feature_bounds[0] + (lower_bound - feature_bounds[0])*p
        
        # Solve Minimization problem to find the new Upperbound.
        sat_minimum = model_min.solve(log_output=False)
        
        # If the Minimization problem is satisfiable.
        if sat_minimum:
            #Update Upperbound
            feature_bounds[1] = feature_bounds[1] + (X_minimum[exclude].solution_value  - precision - feature_bounds[1])*p
            
            # If the updated Upperbound value is lower than the original feature value, undo update.
            if feature_bounds[1] <= data[exclude]:
                feature_bounds[1] = data[exclude]
        
        # If the Minimization problem is unsatisfiable.
        else:
            # Upperbound unchaged
            feature_bounds[1] = feature_bounds[1] + (upper_bound - feature_bounds[1])*p
            
        #Updates checked features list
        checked.append(exclude)
        
        #Checks bound order to assure correct model updating.
        if feature_bounds[0] < feature_bounds[1]:
            features_ranges.append((exclude, [feature_bounds[0], feature_bounds[1]]))
            X_minimum[exclude].lb = feature_bounds[0]
            X_minimum[exclude].ub = feature_bounds[1]
            X_maximum[exclude].lb = feature_bounds[0]
            X_maximum[exclude].ub = feature_bounds[1]
            ranges[exclude] = feature_bounds
        elif feature_bounds[0] > feature_bounds[1]:
            features_ranges.append((exclude, [feature_bounds[1], feature_bounds[0]]))
            X_minimum[exclude].lb = feature_bounds[1]
            X_minimum[exclude].ub = feature_bounds[0]
            X_maximum[exclude].lb = feature_bounds[1]
            X_maximum[exclude].ub = feature_bounds[0]
            ranges[exclude][0] = feature_bounds[1]
            ranges[exclude][1] = feature_bounds[0]
        else:
            features_ranges.append((exclude, [feature_bounds[1], feature_bounds[0]]))
            X_minimum[exclude].lb = feature_bounds[1]
            X_minimum[exclude].ub = feature_bounds[0]
            X_maximum[exclude].lb = feature_bounds[1]
            X_maximum[exclude].ub = feature_bounds[0]
            ranges[exclude][0] = feature_bounds[1]
            ranges[exclude][1] = feature_bounds[0]
    
    #Step 2
    #Take into account ranges found at the last step
    for i in range(len(X_minimum)):
        if i in minimal:
            X_minimum[i].lb = ranges[i][0]
            X_minimum[i].ub = ranges[i][1]
            X_maximum[i].lb = ranges[i][0]
            X_maximum[i].ub = ranges[i][1]
        else:
            X_minimum[i].lb = ranges[i][0]
            X_minimum[i].ub = ranges[i][1]
            X_maximum[i].lb = ranges[i][0]
            X_maximum[i].ub = ranges[i][1]
            
    # For every feature
    for feat_ranges_index, j in enumerate((minimal)):
        #The feature to be checked
        exclude = j
        
        # Set Optimization bounds for the Minimize model
        X_minimum[exclude].lb = ranges[exclude][1] #Upper bound obtained from Step 1
        X_minimum[exclude].ub = upper_bound #Actual upper bound of the feature
        model_min.minimize(X_minimum[exclude])
        
        # Set Optimization bounds for the Maximize model
        X_maximum[exclude].lb = lower_bound #Actual lower bound of the feature
        X_maximum[exclude].ub = ranges[exclude][0]#Lower bound obtained from Step 1
        model_max.maximize(X_maximum[exclude])
        
        # Store feature bounds.
        feature_bounds = [ranges[exclude][0], ranges[exclude][1]]
        
        # Solve Maximization problem
        sat_maximum = model_max.solve(log_output=False)
        
        # If satisfiable.
        if sat_maximum:
            # Update Lowerbound
            feature_bounds[0] =  X_maximum[exclude].solution_value + precision
            
            # If the updated Lowerbound value is higher than the original feature value, undo update.
            if feature_bounds[0] >= ranges[exclude][0]:
                feature_bounds[0] = ranges[exclude][0]
                
        # If unsatisfiable
        else:
            # Lowerbound unchaged
            feature_bounds[0] = lower_bound
        
        # Solve Minimization Problem
        sat_minimum = model_min.solve(log_output=False)
        
        # If satisfiable
        if sat_minimum:
            #Update Upperbound
            feature_bounds[1] = X_minimum[exclude].solution_value - precision
            
            # # If the updated Upperbound value is lower than the original feature value, undo update. 
            if feature_bounds[1] <= ranges[exclude][1]:
                feature_bounds[1] = ranges[exclude][1]
                
        # If unsatisfiable
        else:
            # Upperbound unchaged
            feature_bounds[1] = upper_bound
            
        #Checks bound order to assure correct model updating.    
        if feature_bounds[0] <= feature_bounds[1]:
            features_ranges[feat_ranges_index] = (exclude, [feature_bounds[0], feature_bounds[1]])
            X_minimum[exclude].lb = feature_bounds[0]
            X_minimum[exclude].ub = feature_bounds[1]
            X_maximum[exclude].lb = feature_bounds[0]
            X_maximum[exclude].ub = feature_bounds[1]
            ranges[exclude] = feature_bounds
        else:
            features_ranges[feat_ranges_index] = (exclude, [feature_bounds[1], feature_bounds[0]])
            X_minimum[exclude].lb = feature_bounds[1]
            X_minimum[exclude].ub = feature_bounds[0]
            X_maximum[exclude].lb = feature_bounds[1]
            X_maximum[exclude].ub = feature_bounds[0]
            ranges[exclude][0] = feature_bounds[1]
            ranges[exclude][1] = feature_bounds[0]
        
    return ranges

def onestep(classifier, dual_coef, support_vectors, intercept, data, show_log = 0, lower_bound = 0, upper_bound = 1,
                                       precision = 0.01,noise = 0.0001, threshold = 0, positive=True, problem_name = "Onestep_Explanation"):
    
    if type(data) != np.ndarray:
        try:
            data = np.asarray(data)
        except exception:
            print(f"Data type {type(data)} could not be changed to type numpy.ndarray")   
    if data.ndim >= 2 and len(data) > 1:
        raise UserWarning("More than one pattern has been received. Must pass one pattern only")
        
    # Get minimal explanation    
    minimal = minimal_explanation(
            classifier = classifier,
            dual_coef = dual_coef,
            support_vectors = support_vectors,
            intercept = intercept,
            lower_bound = lower_bound,
            upper_bound = upper_bound,
            data = np.atleast_2d(data),
            positive = positive)[0]
    #print(f'Minimal: {minimal}')
    data = np.ravel(data)
    
    # Create optimization problem/variables
    model_min = docplex.mp.model.Model() # Minimize model
    model_max = docplex.mp.model.Model() # Maximize model
    explanations = [] # List of explanations
    X_minimum = np.asarray([model_min.continuous_var(lower_bound, upper_bound, 'x'+str(i)) for i in range(len(data))]) # Optimization Variables for Minimize model
    X_maximum = np.asarray([model_max.continuous_var(lower_bound, upper_bound, 'x_'+str(i)) for i in range(len(data))]) # Optimization Variables for Maximize model
    
    # R restriction according to c class (Positive or Negative)
    if positive:
        model_min.add_constraint(((dual_coef @ support_vectors) @ X_minimum.reshape(1, len(X_minimum)).T + intercept)[0][0] <= threshold -noise)
        model_max.add_constraint(((dual_coef @ support_vectors) @ X_maximum.reshape(1, len(X_maximum)).T + intercept)[0][0] <= threshold -noise)

    else:
        model_min.add_constraint(((dual_coef @ support_vectors) @ X_minimum.reshape(1, len(X_minimum)).T + intercept)[0][0] >= threshold +noise)
        model_max.add_constraint(((dual_coef @ support_vectors) @ X_maximum.reshape(1, len(X_maximum)).T + intercept)[0][0] >= threshold +noise)
        
    # Get ranges from minimal explanation
    ranges = [] # All feature ranges
    ranges_minimal_index = [] # Store the indexes of relevant features
    for i in range(len(X_minimum)):
        if i in minimal:
            ranges.append([data[i],data[i]])
            ranges_minimal_index.append(i)
            X_minimum[i].lb = data[i]
            X_minimum[i].ub = data[i]
            X_maximum[i].lb = data[i]
            X_maximum[i].ub = data[i]
        else:
            ranges.append([lower_bound, upper_bound])
            X_minimum[i].lb = lower_bound
            X_minimum[i].ub = upper_bound
            X_maximum[i].lb = lower_bound
            X_maximum[i].ub = upper_bound
    
    # Auxiliary Variables        
    checked = [] # Stores checked features
    features_ranges = [] # Stores found explanation ranges
    not_relevant = [] #Stores irrelevant features

    # For every feature    
    for j in (minimal):
        
        # The feature to be checked
        exclude = j
        
        # Set Optimization bounds for the Minimize model
        X_minimum[exclude].lb = data[exclude]  # Original Value
        X_minimum[exclude].ub = upper_bound    # Upperbound
        model_min.minimize(X_minimum[exclude]) # Cost Function

        # Set Optimization bounds for the Maximize model
        X_maximum[exclude].lb = lower_bound    # Lowerbound
        X_maximum[exclude].ub = data[exclude]  # Original Value
        model_max.maximize(X_maximum[exclude]) # Cost Function
        
        # Store feature bounds. Starts with the original value of the feature.
        feature_bounds = [data[exclude], data[exclude]]
        
        # Solve Maximization problem to find the new Lowerbound.
        sat_maximum = model_max.solve(log_output=show_log)
        
        # If the Maximization problem is satisfiable.
        if sat_maximum:
            
            # Update Lowerbound
            feature_bounds[0] =  X_maximum[exclude].solution_value + precision
            #print(f'[Onestep] - feature {exclude} changes with lower value {X_maximum[exclude].solution_value}. Original value {data[exclude]}. New value: {feature_bounds[0]}')
            
            # If the updated Lowerbound value is higher than the original feature value, undo update.
            if feature_bounds[0] >= data[exclude]:
                feature_bounds[0] = data[exclude]
            
        # If the problem is unsatisfiable
        else:         
            # Lowerbound unchaged
            feature_bounds[0] = lower_bound
            #print(f'[Onestep] - feature {exclude} reached lowerbound {lower_bound} without changing the class')
        
        # Solve Minimization problem to find the new Upperbound.
        sat_minimum = model_min.solve(log_output=show_log)
        
        # If the Minimization problem is satisfiable.
        if sat_minimum:   
            #Update Upperbound
            feature_bounds[1] = X_minimum[exclude].solution_value - precision
            #print(f'[Onestep] - feature {exclude} changes with upper value {X_minimum[exclude].solution_value}.\nOriginal value {data[exclude]}. New value: {feature_bounds[1]}')
            
            # If the updated Upperbound value is lower than the original feature value, undo update.
            if feature_bounds[1] <= data[exclude]:
                feature_bounds[1] = data[exclude]
                
        # If the Minimization problem is unsatisfiable.
        else:
            # Upperbound unchaged
            feature_bounds[1] = upper_bound
            #print(f'[Onestep] - feature {exclude} reached upperbound {upper_bound} without changing the class')

        #Updates checked features list
        checked.append(exclude)
        
        #Checks bound order to assure correct model updating.
        if feature_bounds[0] <= feature_bounds[1]:
            features_ranges.append((exclude, [feature_bounds[0], feature_bounds[1]]))
            X_minimum[exclude].lb = feature_bounds[0]
            X_minimum[exclude].ub = feature_bounds[1]
            X_maximum[exclude].lb = feature_bounds[0]
            X_maximum[exclude].ub = feature_bounds[1]
            ranges[exclude] = feature_bounds
            
        else:
            features_ranges.append((exclude, [feature_bounds[1], feature_bounds[0]]))
            X_minimum[exclude].lb = feature_bounds[1]
            X_minimum[exclude].ub = feature_bounds[0]
            X_maximum[exclude].lb = feature_bounds[1]
            X_maximum[exclude].ub = feature_bounds[0]
            ranges[exclude][0] = feature_bounds[1]
            ranges[exclude][1] = feature_bounds[0]
        #print(f'Feature {exclude}, feature ranges: {features_ranges}\n')
    return ranges

def minimal_explanation(classifier, dual_coef, support_vectors, intercept, data, show_log = False, lower_bound = 0, upper_bound = 1,
                                       precision = 0.01,noise = 0.0001, threshold = 0, positive=True, problem_name = "Minimal_Explanation"):
    # Explanations List
    minimal_exps = [] 
    
    # Create optimization problem
    model = docplex.mp.model.Model()
    
    # Create optimization variables
    X = np.asarray([model.continuous_var(lower_bound, upper_bound, 'x'+str(i)) for i in range(len(data[0]))])
    
    # Restriction according to class c
    if positive:
        model.add_constraint(((dual_coef @ support_vectors) @ X.reshape(1, len(X)).T + intercept)[0][0] <= threshold -noise)
    else:
        model.add_constraint(((dual_coef @ support_vectors) @ X.reshape(1, len(X)).T + intercept)[0][0] >= threshold +noise)
        
    
    #For every sample
    for z in range(len(data)):
        relevant_features = []
        features_ranges = []
        not_relevant = []
        minimal = []
        
        #Setting up Problem Variables
        for x in X:
            x.lb = lower_bound
            x.ub = upper_bound
            
        #For every feature    
        for j in range(len(data[z])):
            
            #The feature to be checked
            exclude = j
            
            #Iterate over every feature of the pattern
            for i, feature in enumerate(data[z]):
                
                #If feature is relevant, keep it so that it maintains the class
                if i != exclude and i in relevant_features:
                    X[i].lb = features_ranges[i][0]
                    X[i].ub = features_ranges[i][1]

                #If its not the feature to be checked and haven't been worked upon yet, do not change its value
                elif i != exclude and i not in not_relevant and i not in relevant_features:
                    X[i].lb = feature
                    X[i].ub = feature

                #If feature is the one to be checked or is irrelevant, ranges set to lower/upper bound value    
                elif i == exclude or i in not_relevant:
                    X[i].lb = lower_bound
                    X[i].ub = upper_bound
            
            #Cost function
            #print(model.lp_string)
            model.minimize(X[i])
            
            #Feature value is originally upper/lower limited by the same value
            feature_bounds = [data[z][exclude],data[z][exclude]]
            
            #Check if the feature is relevant and makes the predicted class change
            sat = model.solve(log_output=False)
            
            # If feature is relevant (satisfiable)
            if sat:
                relevant_features.append(exclude)
                features_ranges.append(feature_bounds)
                minimal.append(exclude)
                value = X[exclude].solution_value
                #print(f'[Minimal] {exclude} is relevant with {value}')

            else:
                #Checked feature is not able to change the predicted class.
                not_relevant.append(exclude)
                features_ranges.append([lower_bound, upper_bound])
                #print(f'[Minimal] {exclude} is not relevant')

        minimal_exps.append(minimal)
    return minimal_exps