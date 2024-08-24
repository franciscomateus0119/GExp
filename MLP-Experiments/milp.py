import docplex.mp.model as mp
from cplex import infinity
import numpy as np
import tensorflow as tf
import pandas as pd

# For type annotations
from keras.engine.sequential import Sequential
from keras.layers.core.dense import Dense


def codify_network_fischetti(
        mp_model: mp.Model,
        layers: list[Dense],
        input_variables,
        auxiliary_variables,
        intermediate_variables,
        decision_variables,
        output_variables
):
    output_bounds = []

    for i in range(len(layers)):
        w = layers[i].get_weights()[0].T
        b = layers[i].bias.numpy()
        x = input_variables if i == 0 else intermediate_variables[i - 1]
        if i != len(layers) - 1:
            y = intermediate_variables[i]
        else:
            y = output_variables

        for j in range(w.shape[0]):
            constraint_name = f'c_{i}_{j}'
            if i != len(layers) - 1:
                s = auxiliary_variables[i]
                a = decision_variables[i]
                # w[j, :] @ x + b[j] <= y[j] <==> w[j, :] @ x + b[j] == y[j] - s[j], where s[j] >= 0
                mp_model.add_constraint(w[j, :] @ x + b[j] == y[j] - s[j], constraint_name)
                mp_model.add_indicator(binary_var=a[j], linear_ct=y[j] <= 0, active_value=1)
                mp_model.add_indicator(binary_var=a[j], linear_ct=s[j] <= 0, active_value=0)

                mp_model.maximize(y[j])
                mp_model.solve()
                ub_y = mp_model.solution.get_objective_value()
                mp_model.remove_objective()

                mp_model.maximize(s[j])
                mp_model.solve()
                ub_s = mp_model.solution.get_objective_value()
                mp_model.remove_objective()

                y[j].set_ub(ub_y)
                s[j].set_ub(ub_s)
            else:
                mp_model.add_constraint(w[j, :] @ x + b[j] == y[j], constraint_name)
                mp_model.maximize(y[j])
                mp_model.solve()
                ub = mp_model.solution.get_objective_value()
                mp_model.remove_objective()

                mp_model.minimize(y[j])
                mp_model.solve()
                lb = mp_model.solution.get_objective_value()
                mp_model.remove_objective()

                y[j].set_ub(ub)
                y[j].set_lb(lb)
                output_bounds.append([lb, ub])

    return mp_model, output_bounds


def codify_network_tjeng(
        mp_model: mp.Model,
        layers: list[Dense],
        input_variables,
        intermediate_variables,
        decision_variables,
        output_variables
):
    output_bounds = []

    for i in range(len(layers)):
        w = layers[i].get_weights()[0].T
        b = layers[i].bias.numpy()
        x = input_variables if i == 0 else intermediate_variables[i-1]
        if i != len(layers) - 1:
            y = intermediate_variables[i]
        else:
            y = output_variables

        for j in range(w.shape[0]):
            mp_model.maximize(w[j, :] @ x + b[j])
            mp_model.solve()
            ub = mp_model.solution.get_objective_value()
            mp_model.remove_objective()

            constraint_name = f'c_{i}_{j}'
            if ub <= 0 and i != len(layers) - 1:
                mp_model.add_constraint(y[j] == 0, constraint_name)
                continue

            mp_model.minimize(w[j, :] @ x + b[j])
            mp_model.solve()
            lb = mp_model.solution.get_objective_value()
            mp_model.remove_objective()

            if lb >= 0 and i != len(layers) - 1:
                mp_model.add_constraint(w[j, :] @ x + b[j] == y[j], constraint_name)
                continue

            if i != len(layers) - 1:
                a = decision_variables[i]
                mp_model.add_constraint(y[j] <= w[j, :] @ x + b[j] - lb * (1 - a[j]))
                mp_model.add_constraint(y[j] >= w[j, :] @ x + b[j])
                mp_model.add_constraint(y[j] <= ub * a[j])
            else:
                mp_model.add_constraint(w[j, :] @ x + b[j] == y[j])
                output_bounds.append([lb, ub])

    return mp_model, output_bounds


def codify_network(
        keras_model: Sequential,
        dataframe: pd.DataFrame,
        method: str,
        relax_constraints: bool
):
    layers = keras_model.layers
    num_features = layers[0].get_weights()[0].shape[0]
    mp_model = mp.Model()

    (domain_input, bounds_input_aux) = get_domain_and_bounds_inputs(dataframe)
    bounds_input = np.array(bounds_input_aux)

    if relax_constraints:
        lb = bounds_input[:, 0]
        ub = bounds_input[:, 1]
        input_variables = mp_model.continuous_var_list(num_features, lb, ub, name='x')
    else:
        input_variables = []
        for i in range(len(domain_input)):
            [lb, ub] = bounds_input[i]
            name = f'x_{i}'
            if domain_input[i] == 'C':
                decision_variable = mp_model.continuous_var(lb, ub, name)
                input_variables.append(decision_variable)
            elif domain_input[i] == 'I':
                integer_variable = mp_model.integer_var(lb, ub, name)
                input_variables.append(integer_variable)
            elif domain_input[i] == 'B':
                decision_variable = mp_model.binary_var(name)
                input_variables.append(decision_variable)

    intermediate_variables = []
    auxiliary_variables = []
    decision_variables = []

    for i in range(len(layers) - 1):
        weights: np.ndarray = layers[i].get_weights()[0]
        number_of_variables: int = weights.shape[1]
        key_format = f'_{i}_%s'
        continuous_decision_variables = mp_model.continuous_var_list(number_of_variables, name='y', lb=0,
                                                                     key_format=key_format)
        intermediate_variables.append(continuous_decision_variables)

        if method == 'fischetti':
            continuous_decision_variables = mp_model.continuous_var_list(number_of_variables, name='s', lb=0,
                                                                         key_format=key_format)
            auxiliary_variables.append(continuous_decision_variables)

        if relax_constraints and method == 'tjeng':
            continuous_decision_variables = mp_model.continuous_var_list(number_of_variables, name='a', lb=0, ub=1,
                                                                         key_format=key_format)
            decision_variables.append(continuous_decision_variables)
        else:
            binary_decision_variables = mp_model.binary_var_list(number_of_variables, name='a', lb=0, ub=1,
                                                                 key_format=key_format)
            decision_variables.append(binary_decision_variables)

    output_variables = mp_model.continuous_var_list(layers[-1].get_weights()[0].shape[1], name='o', lb=-infinity)

    if method == 'tjeng':
        (mp_model, output_bounds) = codify_network_tjeng(mp_model, layers, input_variables,
                                                         intermediate_variables, decision_variables,
                                                         output_variables)
    else:
        (mp_model, output_bounds) = codify_network_fischetti(mp_model, layers, input_variables, auxiliary_variables,
                                                             intermediate_variables, decision_variables,
                                                             output_variables)

    if relax_constraints:
        # Tighten domain of variables 'a'
        for i in decision_variables:
            for a in i:
                a.set_vartype('Integer')

        # Tighten domain of input variables
        for i, x in enumerate(input_variables):
            if domain_input[i] == 'I':
                x.set_vartype('Integer')
            elif domain_input[i] == 'B':
                x.set_vartype('Binary')
            elif domain_input[i] == 'C':
                x.set_vartype('Continuous')

    return mp_model, output_bounds


def get_domain_and_bounds_inputs(dataframe: pd.DataFrame) -> tuple[list[str], list[list[float]]]:
    domain: list[str] = []
    bounds: list[list[float]] = []

    for column_label in dataframe.columns[:-1]:
        column = dataframe[column_label]
        if len(column.unique()) == 2:
            domain.append('B')
        elif np.any(column.unique().astype(np.int64) != column.unique().astype(np.float64)):
            domain.append('C')
        else:
            domain.append('I')

        bound_inf: float = column.min()
        bound_sup: float = column.max()
        bounds.append([bound_inf, bound_sup])

    return domain, bounds


def main():
    path_dir = 'glass'
    model = tf.keras.models.load_model(f'datasets/{path_dir}/teste.h5')
    testing_data = pd.read_csv(f'datasets/{path_dir}/test.csv')
    training_data = pd.read_csv(f'datasets/{path_dir}/train.csv')
    data = training_data.append(testing_data)
    data = data[['RI', 'Na', 'target']]
    (mdl, bounds) = codify_network(model, data, 'tjeng', False)
    print(mdl.export_to_string())
    print(bounds)


if __name__ == '__main__':
    main()

# X ---- E
# x1 == 1 /\ x2 == 3 /\ F /\ ~E    UNSATISFIABLE
# x1 >= 0 /\ x1 <= 100 /\ x2 == 3 /\ F /\ ~E    UNSATISFIABLE -> x1 isn't relevant,  SATISFIABLE -> x1 is relevant
