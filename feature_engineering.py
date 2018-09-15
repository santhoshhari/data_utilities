from sklearn.model_selection import KFold


def regularized_target_encoding(train, col, target_col, splits=5):
    """
    Function to perform regularized target mean encoding.

    :param train: training dataframe
    :param col: column name on which target mean encoding is to be performed
    :param target_col: name of the target column
    :param splits: Number of folds to split the data for regularization

    :return: training data with regularized mean encoded features
    """
    kf = KFold(n_splits=splits)
    global_mean = train[target_col].mean()
    for train_index, test_index in kf.split(train):
        kfold_mean = train.iloc[train_index].groupby(col)[target_col].mean()
        train.loc[test_index, col + '_mean_enc'] = train.loc[test_index, col].map(kfold_mean)
    train[col + "_mean_enc"].fillna(global_mean, inplace=True)
    train[col + "_mean_enc"] = train[col + "_mean_enc"].astype('float32')
    return train


def mean_encoding_test(test, train, col, target_col):
    """
    Function to perform target mean encoding on the test data.

    :param test: test data (pandas dataframe)
    :param train: training data (pandas dataframe)
    :param col: column name on which target mean encoding is to be performed
    :param target_col : name of the target column

    :return: test data with regularized mean encoded features
    """
    global_mean = train[target_col].mean()
    mean_col = train.groupby(col)[target_col].mean()
    test[col + "_mean_enc"] = test[col].map(mean_col)
    test[col + "_mean_enc"].fillna(global_mean, inplace=True)
    test[col + "_mean_enc"] = test[col + "_mean_enc"].astype('float32')
    return test


if __name__ == '__main__':
    print('Module can only be imported.')
