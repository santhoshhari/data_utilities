import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, forest
from sklearn.linear_model import LogisticRegression, Ridge
from data_utilities.feature_engineering import *
from tqdm import tqdm, tqdm_notebook
tqdm.monitor_interval = 0


def set_rf_samples(n):
    """
    Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """
    Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))


def train_cats(inp_df):
    """
    Function to change any columns of strings in a panda's dataframe to a column of
       categorical values. This applies the changes inplace.

    :param inp_df: A pandas dataframe. Any columns of strings will be changed to categorical values.
    :return: None
    """
    for col_name, col in inp_df.items():
        if is_string_dtype(col):
            inp_df[col_name] = col.astype('category').cat.as_ordered()


def apply_cats(tst_df, trn_df):
    """
    Function to change any columns of strings in tst_df into categorical variables using trn_df as
       a template for the category codes.

    :param tst_df: A pandas dataframe. Any columns of strings will be changed to categorical values.
    :param trn_df: A pandas dataframe. When creating a category for df, it looks up the
            what the category's code were in trn and makes those the category codes
            for df.
    :return: None
    """

    for n, c in tst_df.items():
        if (n in trn_df.columns) and (trn_df[n].dtype.name == 'category'):
            tst_df[n] = pd.Categorical(c, categories=trn_df[n].cat.categories, ordered=True)


def numericalize(inp_df, col, name):
    """
    Function to changes the column col from a categorical type to it's integer codes (label encoding).

    :param inp_df: A pandas dataframe. inp_df[name] will be filled with the integer codes from col.
    :param col: The column that is to be changed into the categories.
    :param name: Desired column name after label encoding
    :return: DataFrame with label encoded column added
    """
    if not is_numeric_dtype(col):
        inp_df[name] = col.cat.codes + 1
    return inp_df


def train_rf_model(param_dict, Xtrain, Xvalid, Ytrain, Yvalid,
                   metric=roc_auc_score, model_type='classification'):
    """
    Function to train a random forest model with a given set of hyper-parameters

    :param param_dict: Dict of hyper-parameters that are kept constant
    :param Xtrain: Train Data
    :param Xvalid: Validation Data
    :param Ytrain: Train labels
    :param Yvalid: Validation labels
    :param metric: Metric to compute model performance on
    :param model_type: Model type - classification/regression
    :return: Tuned model, train score and validation score computed using the metric
    """
    if model_type == 'classification':
        model = RandomForestClassifier(**param_dict)
        model.fit(Xtrain, Ytrain)
        train_preds = model.predict_proba(Xtrain)
        valid_preds = model.predict_proba(Xvalid)
        train_score = metric(Ytrain, train_preds[:, 1])
        valid_score = metric(Yvalid, valid_preds[:, 1])
    elif model_type == 'regression':
        model = RandomForestRegressor(**param_dict)
        model.fit(Xtrain, Ytrain)
        train_preds = model.predict(Xtrain)
        valid_preds = model.predict(Xvalid)
        train_score = metric(Ytrain, train_preds)
        valid_score = metric(Yvalid, valid_preds)
    else:
        raise ValueError('Incorrect model_type. Accepted values - classification and regression')

    return model, train_score, valid_score


def train_linear_model(param_dict, Xtrain, Xvalid, Ytrain, Yvalid,
                       metric=roc_auc_score, model_type='classification'):
    """
    Function to train a linear model with a given set of hyper-parameters

    :param param_dict: Dict of hyper-parameters that are kept constant
    :param Xtrain: Train Data
    :param Xvalid: Validation Data
    :param Ytrain: Train labels
    :param Yvalid: Validation labels
    :param metric: Metric to compute model performance on
    :param model_type: Model type - classification/regression
    :return: Tuned model, train score and validation score computed using the metric
    """
    if model_type == 'classification':
        model = LogisticRegression(**param_dict)
        model.fit(Xtrain, Ytrain)
        train_preds = model.predict_proba(Xtrain)
        valid_preds = model.predict_proba(Xvalid)
        train_score = metric(Ytrain, train_preds[:, 1])
        valid_score = metric(Yvalid, valid_preds[:, 1])
    elif model_type == 'regression':
        model = Ridge(**param_dict)
        model.fit(Xtrain, Ytrain)
        train_preds = model.predict(Xtrain)
        valid_preds = model.predict(Xvalid)
        train_score = metric(Ytrain, train_preds)
        valid_score = metric(Yvalid, valid_preds)
    else:
        raise ValueError('Incorrect model_type. Accepted values - classification and regression')

    return model, train_score, valid_score


def train_xgb_model(param_dict, Xtrain, Xvalid, Ytrain, Yvalid,
                    metric=roc_auc_score, model_type='classification'):
    """
    Function to train an XGBoost model with a given set of hyper-parameters

    :param param_dict: Dict of hyper-parameters that are kept constant
    :param Xtrain: Train Data
    :param Xvalid: Validation Data
    :param Ytrain: Train labels
    :param Yvalid: Validation labels
    :param metric: Metric to compute model performance on
    :param model_type: Model type - classification/regression
    :return: Tuned model, train score and validation score computed using the metric
    """
    if model_type == 'classification':
        model = XGBClassifier(**param_dict)
        model.fit(Xtrain, Ytrain)
        train_preds = model.predict_proba(Xtrain)
        valid_preds = model.predict_proba(Xvalid)
        train_score = metric(Ytrain, train_preds[:, 1])
        valid_score = metric(Yvalid, valid_preds[:, 1])
    elif model_type == 'regression':
        model = XGBRegressor(**param_dict)
        model.fit(Xtrain, Ytrain)
        train_preds = model.predict(Xtrain)
        valid_preds = model.predict(Xvalid)
        train_score = metric(Ytrain, train_preds)
        valid_score = metric(Yvalid, valid_preds)
    else:
        raise ValueError('Incorrect model_type. Accepted values - classification and regression')

    return model, train_score, valid_score


def train_model_regularized_encoding(train, valid, target_col, params, cat_cols, enc_folds=5,
                                     metric=mean_squared_error, model='xgb', model_type='regression'):
    """
    Function to perform model training with support for regularised mean encoding

    :param train: Input dataset to train model on
    :param valid: Validation dataser
    :param target_col: target column name
    :param params: Set of hyper-parameters over which the model is to be tuned
        sklearn ParameterGrid object
    :param cat_cols: categorical columns for mean encoding
    :param enc_folds: Number of folds to be used for regularized encoding
    :param metric: Metric to evaluate model performance on
    :param model: String indicating the type of model (linear, rf, xgb)
    :param model_type: Type of model, regression or classification
    :return: Trained model, train and validation scores
    """
    train_df = train.copy()
    valid_df = valid.copy()

    train_cats(train_df)
    apply_cats(valid_df, train_df)
    for col in cat_cols:
        train_df = regularized_target_encoding(train_df, col, target_col, splits=enc_folds)
        valid_df = mean_encoding_test(valid_df, train_df, col, target_col)

    Xtrain, Ytrain = train_df.drop(columns=[target_col]), train_df[target_col]
    Xvalid, Yvalid = valid_df.drop(columns=[target_col]), valid_df[target_col]

    del train_df
    del valid_df

    if model == 'linear':
        for n, c in Xtrain.items():
            # Label encode categorical columns with more than 10 levels
            if not is_numeric_dtype(c) and c.nunique() > 10:
                Xtrain = numericalize(Xtrain, c, n)
                Xvalid = numericalize(Xvalid, c, n)
        # One hot encode categorical variables with less than 10 less
        Xtrain = pd.get_dummies(Xtrain, dummy_na=True)
        Xvalid = pd.get_dummies(Xvalid, dummy_na=True)
        # Scale features
        std_sclr = StandardScaler()
        Xtrain = std_sclr.fit_transform(Xtrain)
        Xvalid = std_sclr.transform(Xvalid)
    else:
        # Convert cateforical variables to numeric representations
        for n, c in Xtrain.items():
            Xtrain = numericalize(Xtrain, c, n)
        for n, c in Xvalid.items():
            Xvalid = numericalize(Xvalid, c, n)

    if model == 'xgb':
        return train_xgb_model(params, Xtrain, Xvalid, Ytrain, Yvalid,
                               metric=metric, model_type=model_type)
    elif model == 'rf':
        return train_rf_model(params, Xtrain, Xvalid, Ytrain, Yvalid,
                              metric=metric, model_type=model_type)
    elif model == 'linear':
        return train_linear_model(params, Xtrain, Xvalid, Ytrain, Yvalid,
                                  metric=metric, model_type=model_type)
    else:
        raise ValueError('Incorrect Model, expected rf/xgb/linear')


def train_model_regularized_encoding_cv(train, target_col, param_grid, cat_cols, cv_folds=5,
                                        enc_folds=5, metric=mean_squared_error, model='xgb',
                                        model_type='regression', rf_sample=None):
    """
    Function to perform grid search cross-validation with support for regularised mean encoding

    :param train: Input dataset Pandas DataFrame
    :param target_col: target column name
    :param param_grid: Set of hyper-parameters over which the model is to be tuned
        sklearn ParameterGrid object
    :param cat_cols: categorical columns for mean encoding
    :param cv_folds: Number of folds to be used for cross validation
    :param enc_folds: Number of folds to be used for regularized encoding
    :param metric: Metric to evaluate model performance on
    :param model: String indicating the type of model (linear, rf, xgb)
    :param model_type: Type of model, regression or classification
    :param rf_sample: Number of observations each tree in random forest sees
    :return: DataFrame of the parameters explored and corresponding model performance
    """
    kf = KFold(cv_folds, random_state=42)
    columns = [*param_grid[0].keys()] + ['train_score', 'valid_score']
    results = pd.DataFrame(columns=columns)
    for params in tqdm_notebook(param_grid):
        train_scores = list()
        valid_scores = list()
        for train_idx, test_idx in kf.split(train):
            # Split data into train and test
            kf_train, kf_test = train.iloc[train_idx], train.iloc[test_idx]
            kf_train.reset_index(inplace=True, drop=True)
            kf_test.reset_index(inplace=True, drop=True)

            train_cats(kf_train)
            apply_cats(kf_test, kf_train)

            for col in cat_cols:
                kf_train = regularized_target_encoding(kf_train, col, target_col, splits=enc_folds)
                kf_test = mean_encoding_test(kf_test, kf_train, col, target_col)

            Xtrain, Ytrain = kf_train.drop(columns=[target_col]), kf_train[target_col]
            Xvalid, Yvalid = kf_test.drop(columns=[target_col]), kf_test[target_col]

            if model == 'linear':
                for n, c in Xtrain.items():
                    # Label encode categorical columns with more than 10 levels
                    if not is_numeric_dtype(c) and c.nunique() > 10:
                        Xtrain = numericalize(Xtrain, c, n)
                        Xvalid = numericalize(Xvalid, c, n)
                # One hot encode categorical variables with less than 10 less
                Xtrain = pd.get_dummies(Xtrain, dummy_na=True)
                Xvalid = pd.get_dummies(Xvalid, dummy_na=True)
                # Scale features
                std_sclr = StandardScaler()
                Xtrain = std_sclr.fit_transform(Xtrain)
                Xvalid = std_sclr.transform(Xvalid)
            else:
                # Convert cateforical variables to numeric representations
                for n, c in Xtrain.items():
                    Xtrain = numericalize(Xtrain, c, n)
                for n, c in Xvalid.items():
                    Xvalid = numericalize(Xvalid, c, n)

            if model == 'xgb':
                _, train_score, valid_score = train_xgb_model(params, Xtrain, Xvalid, Ytrain, Yvalid,
                                                              metric=metric, model_type=model_type)
            elif model == 'rf':
                if rf_sample:
                    set_rf_samples(rf_sample)

                _, train_score, valid_score = train_rf_model(params, Xtrain, Xvalid, Ytrain, Yvalid,
                                                                        metric=metric, model_type=model_type)
                reset_rf_samples()
            elif model == 'linear':
                _, train_score, valid_score = train_linear_model(params, Xtrain, Xvalid, Ytrain, Yvalid,
                                                                 metric=metric, model_type=model_type)
            else:
                raise ValueError('Incorrect Model, expected rf/xgb/linear')

            train_scores.append(train_score)
            valid_scores.append(valid_score)

        to_write = params.copy()
        to_write['train_score'] = np.mean(train_scores)
        to_write['valid_score'] = np.mean(valid_scores)
        results = results.append(pd.DataFrame.from_dict(to_write, orient='index').T)

    return results


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
                       Y_train, Y_valid, fn_train=train_xgb_model, maxiters=100,
                       alpha=0.85, beta=1.3, T_0=0.40, update_iters=5):
    """
    Function to perform hyper-parameter search using simulated annealing
    Detailed explanation at https://github.com/santhoshhari/simulated_annealing

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

        all_params = curr_params.copy()
        all_params.update(const_param)
        model, train_score, valid_score = fn_train(all_params, X_train, X_valid, Y_train, Y_valid)

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
