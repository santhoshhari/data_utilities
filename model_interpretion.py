import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.cluster import hierarchy as hc
from pdpbox import pdp
plt.style.use('fivethirtyeight')


def print_confusion_matrix(confusion_matrix, class_names, figsize=(8, 6)):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    Source - https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the output figure,
        the second determining the vertical size. Defaults to (8,6).

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def dendrogram(X_train):
    """
    Function to plot dendrogram to identify collinear features
    :param X_train: Pandas dataframe with features
    :return: Dendrogram plot
    """
    corr = np.round(scipy.stats.spearmanr(X_train).correlation, 4)
    corr_condensed = hc.distance.squareform(1 - corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=(14, 12))
    dendrogram = hc.dendrogram(z, labels=X_train.columns, orientation='left', leaf_font_size=16)
    plt.show()


def get_sample(df,n):
    ''' Takes a sample of the given dataframe
    Input:
        df : Input dataframe
        n = number of records to be sampled
    Output:
        sampled dataframe
    '''
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()


def plot_pdp(model, X_train, feat, clusters=None, feat_name=None):
    """
    Function to plot dependency of target variable on the feature

    :param model: Trained model
    :param X_train: Datafram to get prediction of model from
    :param feat: Feature to plot target dependency for
    :param clusters: Flag to indicate is clusters are needed
    :param feat_name: Feature name to display on plot
    :return: partial dependency plot
    """
    feat_name = feat_name or feat
    x = get_sample(X_train, 1000)
    p = pdp.pdp_isolate(model, x, x.columns, feat, num_grid_points=20)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                        cluster=clusters is not None, n_cluster_centers=clusters)


def plot_pdp_interact(model, X_train, feats):
    """
    Function to plot dependency of target variable on the feature

    :param model: Trained model
    :param X_train: Datafram to get prediction of model from
    :param feats: List (size 2) of feature to plot target dependency for
    :param clusters: Flag to indicate is clusters are needed
    :param feat_name: Feature name to display on plot
    :return: partial dependency plot
    """
    x = get_sample(X_train, 1000)
    p = pdp.pdp_interact(model, x, x.columns, feats)
    return pdp.pdp_interact_plot(p, feats, plot_pdp=True)


if __name__ == '__main__':
    print('Module can only be imported.')
