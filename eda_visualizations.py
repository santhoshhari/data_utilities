import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature(df, col, target, labels=None):
    """
    Function to plot distribution of col and dependance of target on col
        Left - the distribution of samples on the feature
        Right - the dependence of target on the feature

    :param df: Pandas DataFrame with the data to plot
    :param col: Column to visualize
    :param target: Target variable
    :param labels: X and Y axis labels, tuple of len 2

    :return: None
    """
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    if df[col].dtype == 'int64':
        df[col].value_counts().sort_index().plot()
    else:
        # change the categorical variable to category type and order their level by the mean salary
        # in each category
        mean = df.groupby(col)[target].mean()
        df[col] = df[col].astype('category')
        levels = mean.sort_values().index.tolist()
        df[col].cat.reorder_categories(levels, inplace=True)
        df[col].value_counts().plot()
    plt.xticks(rotation=45)
    plt.xlabel(labels[0])
    plt.ylabel('Counts')
    plt.subplot(1, 2, 2)

    if df[col].dtype == 'int64' or col == 'companyId':
        # plot the mean salary for each category and fill between the (mean - std, mean + std)
        mean = df.groupby(col)[target].mean()
        std = df.groupby(col)[target].std()
        mean.plot()
        plt.fill_between(range(len(std.index)), mean.values - std.values, mean.values + std.values, \
                         alpha=0.1)
    else:
        sns.boxplot(x=col, y=target, data=df)

    plt.xticks(rotation=45)
    plt.ylabel(labels[1])
    plt.show()
