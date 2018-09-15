import pandas as pd
from pandas.core.dtypes.common import is_string_dtype,is_numeric_dtype


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


def proc_df(df, y_fld):
    """
    Function that takes a data frame df and splits off the response variable, and
       changes the df into an entirely numeric dataframe.

    :param df: The data frame you that is to be processed.
    :param y_fld: The name of the response variable

    :return: List containing the final dataframe and its y values
    """
    df = df.copy()
    y = df[y_fld].values
    df.drop([y_fld], axis=1, inplace=True)
    for n, c in df.items():
        df = numericalize(df, c, n)
    res = [df, y]
    return res


if __name__ == '__main__':
    print('Module can only be imported.')
