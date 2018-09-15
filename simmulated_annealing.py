from tqdm import tqdm, tqdm_notebook
tqdm.monitor_interval = 0

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor


# Detailed explanation at https://github.com/santhoshhari/simulated_annealing
def train_model(curr_params, param_dict, Xtrain, Xvalid, Ytrain, Yvalid,
                metric=roc_auc_score, model_type='classification'):
    """
    Function to train an XGBoost model with a given set of hyper-parameters

    :param curr_params: Dict of hyper-parameters to sampled values mapping
    :param param_dict: Dict of hyper-parameters that are kept constant
    :param Xtrain: Train Data
    :param Xvalid: Validation Data
    :param Ytrain: Train labels
    :param Yvalid: Validation labels
    :param metric: Metric to compute model performance on
    :param model_type: Model type - classification/regression
    :return: Tuned model, train score and validation score computed using the metric
    """
    params_copy = param_dict.copy()
    params_copy.update(curr_params)
    if model_type == 'classification':
        model = XGBClassifier(**params_copy)
        model.fit(Xtrain, Ytrain)
        train_preds = model.predict_proba(Xtrain)
        valid_preds = model.predict_proba(Xvalid)
        train_score = metric(Ytrain, train_preds[:, 1])
        valid_score = metric(Yvalid, valid_preds[:, 1])
    elif model_type == 'regression':
        model = XGBRegressor(**params_copy)
        model.fit(Xtrain, Ytrain)
        train_preds = model.predict(Xtrain)
        valid_preds = model.predict(Xvalid)
        train_score = metric(Ytrain, train_preds)
        valid_score = metric(Yvalid, valid_preds)
    else:
        raise ValueError('Incorrect model_type. Accepted values - classification and regression')

    return model, train_score, valid_score


def choose_params(param_dict, curr_params=None):
    """
    Function to choose parameters for next iteration

    :param param_dict: Ordered dictionary of hyper-parameter search space
    :param curr_params: Dict of current hyper-parameters
    :return: Dictionary of sampled parameters
    """
    if curr_params:
        next_params = curr_params.copy()
        param_to_update = np.random.choice(list(param_dict.keys()))
        param_vals = param_dict[param_to_update]
        curr_index = param_vals.index(curr_params[param_to_update])
        if curr_index == 0:
            next_params[param_to_update] = param_vals[1]
        elif curr_index == len(param_vals) - 1:
            next_params[param_to_update] = param_vals[curr_index - 1]
        else:
            next_params[param_to_update] = \
                param_vals[curr_index + np.random.choice([-1, 1])]
    else:
        next_params = dict()
        for k, v in param_dict.items():
            next_params[k] = np.random.choice(v)

    return next_params


def simulate_annealing(param_dict, const_param, X_train, X_valid,
                       Y_train, Y_valid, fn_train, maxiters=100,
                       alpha=0.85, beta=1.3, T_0=0.40, update_iters=5):
    """
    Function to perform hyperparameter search using simulated annealing

    :param param_dict: Ordered dictionary of hyper-parameter search space
    :param const_param: Static parameters of the model
    :param X_train: Train Data
    :param X_valid: Validation Data
    :param Y_train: Train labels
    :param Y_valid: Validation labels
    :param fn_train: Function to train the model
        (Should return model and metric value as tuple)
    :param maxiters: Number of iterations to perform the parameter search for
    :param alpha: factor to reduce temperature
    :param beta: constant in probability estimate
    :param T_0: Initial temperature
    :param update_iters: # of iterations required to update temperature
    :return: DataFrame of the parameters explored and corresponding model performance
    """
    columns = [*param_dict.keys()] + ['Metric', 'Best Metric', 'Train Metric']
    results = pd.DataFrame(index=range(maxiters), columns=columns)
    best_metric = -1.
    prev_metric = -1.
    prev_params = None
    best_params = dict()
    weights = list(map(lambda x: 10 ** x, list(range(len(param_dict)))))
    hash_values = set()
    T = T_0

    for i in tqdm_notebook(range(maxiters)):
        while True:
            curr_params = choose_params(param_dict, prev_params)
            indices = [param_dict[k].index(v) for k, v in curr_params.items()]
            hash_val = sum([i * j for (i, j) in zip(weights, indices)])
            if hash_val in hash_values:
                tqdm.write('Combination revisited')
            else:
                hash_values.add(hash_val)
                break

        model, train_score, valid_score = fn_train(
            curr_params, const_param, X_train, X_valid, Y_train, Y_valid)

        if valid_score > prev_metric:
            tqdm.write(f'Local Improvement in metric from {prev_metric:.4} to {valid_score:.4}' \
                       + ' - parameters accepted')
            prev_params = curr_params.copy()
            prev_metric = valid_score

            if valid_score > best_metric:
                tqdm.write(f'Global improvement in metric from {best_metric:.4f} to {valid_score:.4}' \
                           + ' - best parameters updated')
                best_metric = valid_score
                best_params = curr_params.copy()
                best_model = model
        else:
            rnd = np.random.uniform()
            diff = valid_score - prev_metric
            threshold = np.exp(beta * diff / T)
            if rnd < threshold:
                tqdm.write('No Improvement but parameters accepted. Metric change: ' +
                           f'{diff:.4} threshold: {threshold:.4} random number: {rnd:.4}')
                prev_metric = valid_score
                prev_params = curr_params
            else:
                tqdm.write('No Improvement and parameters rejected. Metric change: ' +
                           f'{diff:.4} threshold: {threshold:.4} random number: {rnd:.4}')

        results.loc[i, list(curr_params.keys())] = list(curr_params.values())
        results.loc[i, 'Metric'] = valid_score
        results.loc[i, 'Best Metric'] = best_metric
        results.loc[i, 'Train Metric'] = train_score

        if i % update_iters == 0:
            T = alpha * T

    return results, best_model, best_params


if __name__ == '__main__':
    print('Module can only be imported.')
