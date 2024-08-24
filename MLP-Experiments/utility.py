import numpy as np
import pandas as pd

def check_targets(original_set):
    
    """
    ## Check if original binary targets are following the [-1, 1] pattern.
    """
    
    original_unique = np.unique(original_set)
    print("Original Targets: ",original_unique,"\nDesired Targets: [-1,1]")
    print("Is original the desired [-1, 1]? ", np.array_equiv(original_unique,np.array([-1,1])))
    if not np.array_equiv(original_unique,np.array([-1,1])):
        if 1 in original_unique:
            print("1 exists in dataset")
            new = np.select([original_set == original_unique[0]],[-1],original_set)
        elif -1 in original_unique:
            print("-1 exists in dataset")
            new = np.select([original_set == original_unique[1]],[1],original_set)
        else:
            print("Neither exists in dataset")
            new = np.select([original_set == original_unique[0],original_set == original_unique[1]],[-1,1],original_set)
        print("New dataset targets consists of: ",np.unique(new))
        return new

def check_targets_0_1(original_set):
    
    """
    ## Check if original binary targets are following the [-1, 1] pattern.
    """
    
    original_unique = np.unique(original_set)
    print("Original Targets: ",original_unique,"\nDesired Targets: [0,1]")
    print("Is original the desired [0, 1]? ", np.array_equiv(original_unique,np.array([0,1])))
    if np.array_equiv(original_unique,np.array([0,1])):
        return original_set
    else:
        if 1 in original_unique:
            print("1 exists in dataset")
            new = np.select([original_set == original_unique[0]],[-1],original_set)
        elif 0 in original_unique:
            print("0 exists in dataset")
            new = np.select([original_set == original_unique[1]],[1],original_set)
        else:
            print("Neither exists in dataset")
            new = np.select([original_set == original_unique[0],original_set == original_unique[1]],[0,1],original_set)
        print("New dataset targets consists of: ",np.unique(new))
        return new
    
def detail_exp(explanations = None, patterns = None, number_of_features=None, feature_names = None, show_explanation = True, show_frequency = True, return_frequency = True, low_val=0, upp_val=1):
    """
    ## Details the explanations obtained from the solver.
    """
    
    if explanations is None:
        raise UserWarning("explanations is None. Must pass the generated explanations.")
    if patterns is None:
        raise UserWarning("patterns is None. Must pass the patterns.")
    if number_of_features is None:
        raise UserWarning("number_of_features is None. Must pass the number of features.")
    
    if feature_names is not None:
        columns_names = [feature_names]
        relevance_df  = pd.DataFrame(columns=columns_names)
    else:
        columns_names = ['x'+str(i) for i in range(number_of_features)]
        relevance_df  = pd.DataFrame(columns=columns_names)
    for i,explanation in enumerate(explanations):
        pattern_row = [0] * number_of_features
        if show_explanation:
            print(f"\nExplanation for Pattern {i} {patterns[i]}")
        if feature_names is None:
            counter = 0
            for lower_val, upper_val in explanation:
                if show_explanation:
                    print(f"{lower_val} <= f{counter} <= {upper_val}")
                if lower_val == low_val and upper_val == upp_val:
                    pattern_row[counter] = 0
                else:
                    pattern_row[counter] = 1
                counter +=1
        else:
            counter = 0
            for lower_val, upper_val in explanation:
                if show_explanation:
                    print(f"{lower_val} <= {feature_names[counter]} <= {upper_val}")
                if lower_val == low_val and upper_val == upp_val:
                    pattern_row[counter] = 0
                else:
                    pattern_row[counter] = 1
                counter +=1
        relevance_df.loc[len(relevance_df), :] = pattern_row
    if return_frequency:
        if show_frequency:
            print(f"\nFrequency: \n{relevance_df.sum()}")
        return relevance_df
        
def find_indexes(classifier, data, threshold=0):
    
    """
    ## Finds the predicted classes based on the given thresholds.
    """
    
    decfun = classifier.decision_function(data)
    #Find the index for all samples classified as POSITIVE (+1 class)
    positive_indexes = np.where(decfun > threshold)[0]
    
    #Find the index for all samples classified as NEGATIVE (-1 class)
    negative_indexes = np.where(decfun < threshold)[0]

    return positive_indexes,negative_indexes
