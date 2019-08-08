import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def plot_boundaries(X_train, y_train, score=None, probability_func=None, support_vectors=None,n_colors = 100, mesh_res = 1000, ax = None):
    X = X_train
    if len(y_train.shape) == 2 and y_train.shape[1] == 1:
        y_train = y_train.reshape(-1)
    margin_x = (X[:, 0].max() - X[:, 0].min())*0.05
    margin_y = (X[:, 1].max() - X[:, 1].min())*0.05
    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y
    hx = (x_max-x_min)/mesh_res
    hy = (y_max-y_min)/mesh_res
    x_domain = np.arange(x_min, x_max, hx)
    y_domain = np.arange(y_min, y_max, hy)
    xx, yy = np.meshgrid(x_domain, y_domain)

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if probability_func is not None:
        Z = probability_func(np.c_[xx.ravel(), yy.ravel()])
        #Z = Z_aux[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
    
        cf = ax.contourf(xx, yy, Z, n_colors, vmin=0., vmax=1., cmap=cm, alpha=.8)
        #plt.colorbar(cf, ax=ax)
        #plt.colorbar(Z,ax=ax)

        #boundary_line = np.where(np.abs(Z-0.5)<0.001)

        #ax.scatter(x_domain[boundary_line[1]], y_domain[boundary_line[0]], color='k', alpha=0.5, s=1)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.text(xx.max() - .3, yy.min() + .3, score,
                size=20, horizontalalignment='right')

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k', s=60, marker='o')
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], c="y", s=10, marker='o')
    