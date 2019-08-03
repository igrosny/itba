import math
import numpy as np

def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

def get_polynimial_set(X, degree = 12, bias = False):
    # Recibe el dataset X de numero_de_muestras x features  y devuelve una matriz con todas las combinaciones 
    # De los productos del grado indicado en degree
    k = 2
    n = degree + k
    pos = 0
    X_mat = np.zeros((X.shape[0],nCr(n,k)))
    for i in range(degree + 1):
        for j in range(i+1):
            X_mat[:,pos] = (X[:,0]**(i-j))*X[:,1]**j
            pos = pos + 1
    if bias:
        return X_mat
    else:
        return X_mat[:,1:]

import keras
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from IPython.display import clear_output

class PlotBoundary(keras.callbacks.Callback):
    def plotBoundary(self):
        clear_output(wait=True)
        fig=plt.figure(figsize=(20,8))
        gs=GridSpec(2,2) # 2 rows, 3 columns
        ax=fig.add_subplot(gs[:,0]) # Second row, span all columns
        axLoss=fig.add_subplot(gs[0,1]) # First row, first column
        axAcc=fig.add_subplot(gs[1,1]) # First row, second column
        #self.fig, (self.ax, self.axLoss, self.axAcc )= plt.subplots(1,3, figsize=(20,4))
        ax.scatter(self.class_1[:,0], self.class_1[:,1], color='b', s=2, alpha=0.5)
        ax.scatter(self.class_0[:,0], self.class_0[:,1], color='r', s=2, alpha=0.5)
        Z = 1 - self.model.predict_proba(get_polynimial_set(np.c_[self.X.flatten(), self.Y.flatten()], self.degree))[:, 0]
        Z = Z.reshape(self.Z_shape)
        ax.contour(self.X, self.Y, Z, (0.5,), colors='k', linewidths=0.5)
        axAcc.plot(self.acc, label="Train Accuracy")
        axAcc.plot(self.val_acc, label="CV Accuracy")
        axLoss.plot(self.loss, label="Train Loss")
        axLoss.plot(self.val_loss, label="CV Loss")
        axAcc.legend()
        axLoss.legend()
        #self.fig.canvas.draw()
        plt.show()
        
        
    def __init__(self, data, labels, degree=1, plots_every_batches=100, N = 300):
        self.plots_every_batches = plots_every_batches
        self.N = N
        self.degree = degree
        mins = data[:,:2].min(axis=0)
        maxs = data[:,:2].max(axis=0)
        X_lin = np.linspace(mins[0], maxs[0], self.N)
        Y_lin = np.linspace(mins[1], maxs[1], self.N)
        self.X, self.Y = np.meshgrid(X_lin, Y_lin)
        self.Z_shape = self.X.shape
        self.acc = []
        self.loss = []
        self.val_acc = []
        self.val_loss = []
        self.class_1 = data[labels == 1]
        self.class_0 = data[labels == 0]
        #ax.set_ylabel('Alturas [cms]')
        #ax.set_xlabel('Pesos [kgs]')
        #plt.colorbar(cf, ax=ax)
        
    def on_train_begin(self, logs={}):
        self.plotBoundary()
        return
    
    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.plotBoundary()
        return
    
    def on_batch_end(self, batch, logs={}):
        #if batch%self.plots_every_batches == 0:
        #    self.acc.append(logs.get('acc'))
        #    self.loss.append(logs.get('loss'))
        #    self.val_acc.append(logs.get('val_acc'))
        #    self.val_loss.append(logs.get('val_loss'))
        #    self.plotBoundary()
        return