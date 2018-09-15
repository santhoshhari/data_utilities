import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy import issubsctype, int


def plot_continuous_feature(df, col, target, labels, format_labels=True):
    """
    Function to plot distribution of continuous feature and its relationship with target
        Left - the distribution of samples on the feature
        Right - the dependence of target on the feature

    :param df: Input dataframe with the required columns
    :param col: Column with data to visualize
    :param target: Target column
    :param labels: Axis labels, tuple of length 2
    :param format_labels: Flag to indicate if y-axis tick labels should be formatted
    :return: matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # plot distribution of col values
    sns.distplot(df[col], ax=ax1)
    ax1.set(
        xlabel=labels[0],
        ylabel='Density',
        title=f'Distribution of {labels[0]}')
    ax1.get_yaxis().set_major_formatter(mpl.ticker.FormatStrFormatter('%.1e'))

    # Visualize relationship between col and target
    sns.regplot(x=col, y=target, data=df, ax=ax2, scatter_kws={'alpha': 0.4})
    ax2.set(
        xlabel=labels[0],
        ylabel=labels[1],
        title=f'{labels[0]} vs {labels[1]}')

    # Format axis tick labels
    if format_labels:
        mpl_formatter = mpl.ticker.FuncFormatter(
            lambda x, p: format(int(x), ','))
        ax1.get_xaxis().set_major_formatter(mpl_formatter)
        ax2.get_xaxis().set_major_formatter(mpl_formatter)
        ax2.get_yaxis().set_major_formatter(mpl_formatter)

    return fig


def plot_categorical_feature(df, col, target, labels, format_labels=True):
    """
    Function to plot distribution of categorical feature and its relationship with target
        Left - the distribution of the feature
        Right - the dependence of target on the feature

    :param df: Input dataframe with the required columns
    :param col: Column with data to visualize
    :param target: Target column
    :param labels: Axis labels, tuple of length 2
    :param format_labels: Flag to indicate if y-axis tick labels should be formatted
    :return: matplotlib figure object
    """
    # create temporary dataframe for plotting
    temp_df = df[[col, target]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # plot distribution of
    sns.lineplot(
        x=range(temp_df[col].nunique()),
        y=col,
        data=temp_df[col].value_counts().sort_values(
            ascending=False).reset_index(),
        ax=ax1)
    ax1.set(
        xlabel=None,
        ylabel='Counts',
        title=f'Distribution of {labels[0]}')
    mpl_formatter = mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax1.get_yaxis().set_major_formatter(mpl_formatter)

    mean = temp_df.groupby(col)[target].mean()
    levels = mean.sort_values().index.tolist()

    if len(levels) < 11:
        temp_df[col] = pd.Categorical(
            temp_df[col], categories=levels, ordered=True)
        sns.boxplot(x=col, y=target, data=temp_df, ax=ax2)
        ax2.set(
            xlabel=labels[0],
            ylabel=labels[1],
            title=f'{labels[0]} vs {labels[1]}')
        if format_labels:
            ax2.get_yaxis().set_major_formatter(mpl_formatter)
    else:
        if not issubsctype(temp_df[col], int):
            temp_df[col] = temp_df[col].astype('category').cat.as_ordered()
            temp_df[col] = temp_df[col].cat.codes + 1

        sns.regplot(x=col, y=target, data=temp_df, ax=ax2, scatter_kws={'alpha': 0.4})
        ax2.set(
            xlabel=labels[0],
            ylabel=labels[1],
            title=f'{labels[0]} vs {labels[1]}')

        if format_labels:
            ax2.get_xaxis().set_major_formatter(mpl_formatter)
            ax2.get_yaxis().set_major_formatter(mpl_formatter)

    return fig


def target_mean_encode(df, cols, target_col):
    """
    Function to label encode cols. Acts inplace.

    :param df: Input dataframe with the required columns
    :param cols: List of columns to be encode
    :param target_col: Target column
    :return: None
    """
    global_mean = df[target_col].mean()
    for col in cols:
        mean_col = df.groupby(col)[target_col].mean()
        df[col] = df[col].map(mean_col)
        df[col].fillna(global_mean, inplace=True)
    return


def plot_correlations(df, target, numeric_cols, cat_columns=None):
    """
    Function to plot heatmap of pairwise correlations of the numeric
    and categorical columns given. Categorical columns are label encoded
    before computation of correlation

    :param df: Input dataframe with the required columns
    :param target: Target column
    :param numeric_cols: List of numeric columns to be considered
    :param cat_columns:  List of categorical columns to be considered
    :return: matplotlib figure object
    """
    if cat_columns:
        temp_df = df[numeric_cols + cat_columns + [target]]
        target_mean_encode(temp_df, cat_columns, target)

    else:
        temp_df = df[numeric_cols + [target]]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        temp_df.corr(),
        vmin=-1,
        vmax=1,
        cmap="RdYlGn",
        annot=True,
        fmt='.2f',
        ax=ax)
    ax.set(title='Correlation Matrix')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    return fig


if __name__ == '__main__':
    print('Module can only be imported.')
