#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A collection of functions for plotting the output of classifiers and their decision boundaries.


'''
import matplotlib.pyplot as pl
import numpy as np

def plot_dataset(x, y, s=20):
    '''Plots a two-dimensional dataset with one or several classes.
    Class is assumed to be an integer value (needed for color) and of dimension (rows,)
    x is assumed to have dimension (rows, 2)
    s (default 20) is desired size of the point. Accepts a scalar or an array of equal length as x and y
    to scale each point individually.
    '''
    fig = pl.figure()
    ax = pl.subplot(111)

    ax.scatter(x[:, 0], x[:, 1], marker='o', c=y, s=s)

    return fig

def plot_boundaries(x, y, model, s=40):
    '''Plot the decision boundaries of a model for a two dimensional dataset.
    The model is expected to have a method with the signature:
        predict(x)
    x being of dimension (rows, 2) in this case.

    y is assumed to contain integer values denoting the class (for colormapping purposes)

    s is optional and denotes the size of the points
    '''
    # create a mesh to plot in
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    fig = pl.figure()
    ax = pl.subplot(111)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #Color the background along the mesh
    pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
    #Color the outlines
    pl.contour(xx, yy, Z, colors='k')
    pl.axis('off')

    #Plot the individual points also
    pl.scatter(x[:, 0], x[:, 1], c=y, s=s, cmap=pl.cm.Paired)

    return fig

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn import svm
    from util import show

    ##DATASETS##
    x, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, n_classes=3)

    plot_dataset(x, y, 30)

    ##BOUNDARIES##
    rbf_svc = svm.SVC(kernel='rbf').fit(x, y)

    plot_boundaries(x, y, rbf_svc)


    show()
