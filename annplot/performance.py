#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Functions for plotting the performance of models, for example:
Area under the Receiving Operating Characteristic (ROC curve)
Training/Validation performance over time
Regression performance
Variable importance


'''
from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_train_vs_test(param, train_perf, test_perf,
                       xlabel="Regularization parameter"):
    '''Plot the performance on the test set and test set as a function of a
    parameter which could be a regression parameter or just the epoch number
    for example.
    You can set the label on the x-axis with the keyword argument xlabel.

    Otherwise it expects parameters of the SAME dimensions, namely (rows,)
    '''
    fig = pl.figure()
    ax = pl.subplot(1, 1, 1)
    ax.semilogx(param, train_perf, label='Train')
    ax.semilogx(param, test_perf, label='Test')

    optimum = test_perf.argmax()

    #Plot line at optimum
    ax.vlines(param[optimum], ax.get_ylim()[0], test_perf[optimum],
                        color='k', linewidth=3, label='Optimum on test')
    pl.legend(loc='best')
    pl.ylim([ax.get_ylim()[0], 1.2 * ax.get_ylim()[1]])
    pl.xlabel(xlabel)
    pl.ylabel('Performance')

    return fig


def plot_roc(targets, outputs, labels=None):
    '''Plot the ROC curve for several models.
    Optionally, use the keyword labels to
    add labels for each model but make sure to specify one label
    for each model you have.

    Examples:
    Y_true = [1,1,0,0]
    pred1 = [0.9, 0.8, 0.6, 0.4]
    pred2 = [0.8, 0.3, 0.5, 0.4]

    plot_roc(Y_true, [pred1])

    labels = ["Good model", "Bad model"]

    plot_roc(Y_true, [pred1, pred2], labels=labels)

    show()
    '''
    if not labels:
        labels = ["" for each in outputs]

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    fig = pl.figure()
    ax = pl.subplot(1, 1, 1)

    for output, label in zip(outputs, labels):
        fpr, tpr, thresholds = roc_curve(targets, output)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1,
                label='{} (area = {:0.2f})'.format(label, roc_auc))

    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(outputs)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    if len(outputs) > 1:
        ax.plot(mean_fpr, mean_tpr, 'k--',
                label='Mean ROC (area = {:0.2f})'.format(mean_auc), lw=2)

    pl.xlim([-0.02, 1.02])
    pl.ylim([-0.03, 1.03])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    #pl.title('Receiver operating characteristic')
    pl.legend(loc="lower right")

    return fig


def plot_feature_importance(names, feature_changes):
    '''Gives a graph of the relative importance among features sorted
    by impact.
    Expects the feature_changes to consist of the change in the error function
    when that feature was removed in some way

    With names being the names you want to print along with that feature.
    '''

    #Sort by most important and make relative to the biggest change
    feature_changes = 100 * (feature_changes / feature_changes.max())
    sorted_idx = np.argsort(feature_changes)

    pos = np.arange(sorted_idx.shape[0]) + 0.5

    fig = pl.figure()
    ax = pl.subplot(111)

    ax.barh(pos, feature_changes[sorted_idx], align='center')
    pl.yticks(pos, names[sorted_idx])
    pl.xlabel('Relative Importance')

    return fig

if __name__ == '__main__':
    from util import show, save
    ###ROC###
    Y_true = [1, 1, 0, 0]
    pred1 = [0.9, 0.8, 0.6, 0.4]
    pred2 = [0.8, 0.3, 0.5, 0.4]

    plot_roc(Y_true, [pred1])

    labels = ["Good model", "Bad model"]

    save(plot_roc(Y_true, [pred1, pred2], labels=labels), "roc.png")

    ###TRAIN_VS_TEST###
    from sklearn import linear_model

    ###########################################################################
    # Generate sample data
    n_samples_train, n_samples_test, n_features = 75, 150, 500
    #np.random.seed(0)
    coef = np.random.randn(n_features)
    coef[50:] = 0.0  # only the top 10 features are impacting the model
    X = np.random.randn(n_samples_train + n_samples_test, n_features)
    y = np.dot(X, coef)

    # Split train and test data
    X_train, X_test = X[:n_samples_train], X[n_samples_train:]
    y_train, y_test = y[:n_samples_train], y[n_samples_train:]

###############################################################################
    # Compute train and test errors
    alphas = np.logspace(-5, 1, 60)
    enet = linear_model.ElasticNet(rho=0.7)
    train_errors = list()
    test_errors = list()
    for alpha in alphas:
        enet.set_params(alpha=alpha)
        enet.fit(X_train, y_train)
        train_errors.append(enet.score(X_train, y_train))
        test_errors.append(enet.score(X_test, y_test))

    i_alpha_optim = np.argmax(test_errors)
    alpha_optim = alphas[i_alpha_optim]
    print "Optimal regularization parameter : %s" % alpha_optim

    ###PLOTIT###
    plot_train_vs_test(alphas, np.array(train_errors),
                       np.array(test_errors), "Alpha")
    save(plot_train_vs_test(alphas, np.array(train_errors),
                            np.array(test_errors)), "perf.png")

    ###FEATURE IMPORTANCE###
    from sklearn import ensemble
    from sklearn import datasets
    from sklearn.utils import shuffle
    from sklearn.metrics import mean_squared_error

    ###########################################################################
    # Load data
    boston = datasets.load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

###############################################################################
# Fit regression model
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
              'learn_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)

    ###PLOTIT###
    plot_feature_importance(boston.feature_names, clf.feature_importances_)

    ###Show them###
    show()
