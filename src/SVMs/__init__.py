"""
In this module the data for SVMs is generated and local variables declared
"""
import os

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

from definitions import SEED, ROOT_DIR

NUMBER_OF_SAMPLES = 10

X, Y = make_moons(NUMBER_OF_SAMPLES, random_state=SEED, noise=0.34)

x_axis_limit = [-3, 3]
y_axis_limit = [-2.5, 2.5]


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    print('Here?')
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    print('here?')
    Z = Z.reshape(xx.shape)
    print('h?')
    out = ax.contourf(xx, yy, Z, **params)
    return out


def save_decision_boundary_image(X, Y, filename: str, fitted_model):
    fig, ax = plt.subplots()
    title = 'Decision Boundary'
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)
    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    plot_contours(ax, fitted_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=20, marker='x')
    ax.set_title(title)
    fig.savefig(filename)


def save_plot_data(X, Y, title, filename):
    fig, ax = plt.subplots()
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, marker='x')
    ax.set_title(title)
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)
    fig.savefig(filename)


if not os.path.isdir(ROOT_DIR + '/figures'):
    os.mkdir(ROOT_DIR + '/figures')

save_plot_data(X, Y, 'Dataset for SVMs', ROOT_DIR + '/figures/SVM_Data.png')
