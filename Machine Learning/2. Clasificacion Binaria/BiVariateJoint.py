import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

def get_class_prob_naive(x_data, y_data, joint_class_1, joint_class_2, likelihood_indep_class_1, likelihood_indep_class_2):
    prior_class_1 = joint_class_1.N/ (joint_class_1.N + joint_class_2.N)
    prior_class_2 = joint_class_2.N/ (joint_class_1.N + joint_class_2.N)
    likelihood_class_1 = likelihood_indep_class_1[joint_class_1.data_to_index(x_data, y_data)] 
    likelihood_class_2 = likelihood_indep_class_2[joint_class_2.data_to_index(x_data, y_data)]
    total = likelihood_class_1*prior_class_1 + prior_class_2*likelihood_class_2
    # Evita division por cero
    total[total==0] = 1
    p_class_1 = prior_class_1*likelihood_class_1/total
    p_class_2 = prior_class_2*likelihood_class_2/total
    # Las indeterminadas en 0.5
    p_class_1[total==1] = 0.5
    p_class_2[total==1] = 0.5
    return p_class_1, p_class_2

def get_class_prob(x1_data, x2_data, joint_class_1, joint_class_2):
    '''
    Devuelve algo
    
    Args:
        x1_data (int): Los datos de la primera clase
        x2_data (int): los datos de la segunda clase
        joint_class_1 (BiVariateJoint): Distribucion de primera clase
        joint_class_2 (BiVariateJoint): Distribucion de segunda clase
    
    Returns:
        p_class_1 
        p_class_2
    '''
    
    # Calculo la prior de cada clase
    prior_class_1 = joint_class_1.N / (joint_class_1.N + joint_class_2.N)
    prior_class_2 = joint_class_2.N / (joint_class_1.N + joint_class_2.N)
    
    likelihood_class_1 = joint_class_1.get_prob(x1_data, x2_data)
    likelihood_class_2 = joint_class_2.get_prob(x1_data, x2_data)
    
    # Calculo las marginal
    total = likelihood_class_1 * prior_class_1 + prior_class_2 * likelihood_class_2
    # Evita division por cero
    total[total==0] = 1
    
    p_class_1 = prior_class_1 * likelihood_class_1 / total
    p_class_2 = prior_class_2 * likelihood_class_2 / total
    
    # Las indeterminadas en 0.5
    p_class_1[total==1] = 0.5
    p_class_2[total==1] = 0.5
    
    return p_class_1, p_class_2                

class BiVariateJoint:
    '''
    Clase para generar una distribucion bivariada

    
    Attributes:
        data (np.array): Tiene que ser np.array de dos columnas
        step_X (int): de a cuanto quiero agrupar los valores de X (default 1)
        step_Y (int): de a cuanto quiero agrupar los valores de Y (default 1)
        data_rounded (np.array): Son los valores redondeados, los divido por el step, redondeo y multiplico
        maxs (np.array): Los valores mas altos de cada columna
        mins (np.array): Los valores mas chicos de cada columna
        frequencies:
        X
        Y
        joint_matrix
        N (int): La cantidad de conjunto de datos
    '''
    def __init__(self, data, step_X = 1, step_Y = 1, mins=None, maxs=None):
        '''
        Constructor
        
        Args:
            data (int)
            step_X
            step_Y
            mins
            maxs
        
        Returns:
            null
        '''
        self.step_X = step_X
        self.step_Y = step_Y
        step = np.array([step_X, step_Y])
        self.data = data
        self.data_rounded = (np.round(data / step) * step)
        
        # Si no le paso los maximos los calcula
        if maxs is None:
            self.maxs = np.max(self.data_rounded, axis = 0) + 1
        else:
            self.maxs = maxs
        
        # Si no le paso los minimos los calcula
        if mins is None:
            self.mins = np.min(self.data_rounded, axis = 0) - 1
        else:
            self.mins = mins
        
        # Lo convierto a tuplas y cuento la cantidad de cada uno
        # Y cuento cuantos epsacios tengo de cada uno
        tuples = [tuple(row) for row in self.data_rounded]        
        self.frequencies = Counter(tuples)
        
        # Agrego uno adelante y otro atras para cubrirme
        count_X = int(np.round((self.maxs[0] - self.mins[0]) / step_X)) + 1
        count_Y = int(np.round((self.maxs[1] - self.mins[1]) / step_Y)) + 1

        # np.linspace: Devuelve numero espeaciados de manera uniforme sobre un intervalo definido
        self.X = np.linspace(self.mins[0] - step_X, self.mins[0] + step_X * count_X, count_X + 2)
        self.Y = np.linspace(self.mins[1] - step_Y, self.mins[1] + step_Y * count_Y, count_Y + 2)
        
        # Crea la matriz de frecuencia
        self.joint_matrix = self.freq_2_matrix()
        
        # Guarda el tamaño del arreglo
        self.N = len(data)
    
    def plot_data(self, color='b'):
        plt.scatter(self.data[:,0], self.data[:,1], color=color, s=2)
    
    def plot_rounded(self, color='b'):
        plt.scatter(self.data_rounded[:,0], self.data_rounded[:,1], color=color, s=2)
    
    def data_to_index(self, x, y):
        '''
        Funcion que convierte un par de variables en los indices de las matriz
        
        Args:
            x (int): valor de la priemr variables
            y (int): valor de la segunda variable
        
        Returns:
            x, y (int, int): los indices de la matriz
        '''
        x = np.round((x - self.X[0])/self.step_X).astype(int)
        y = np.round((y - self.Y[0])/self.step_Y).astype(int)
        return x, y

    def get_prob(self, x, y, normalized=True):
        '''
        Devuelve la probabilidad conjunta de dos variables, con la opcion de 
        ser normalizadas
        
        Args:
            x (int): valor de la primera vairable
            y (int): valor de la segunda variable
            normalized (boolean): true, si quiero que me devuelva el valor normalizado
            
        Return
            prob: la probabilidad conjunta de las variables x, y
        '''
        x, y = self.data_to_index(x, y)

        if normalized:
            prob = self.joint_matrix[x , y]/self.N
        else:
            prob = self.joint_matrix[x , y]
            
        return prob
    
    def freq_2_matrix(self):
        '''
        Crea la matriz de frecuencia conjunta con los valores
        
        Returns:
            joint (np.array): La matriz de frecuencia
        '''
        joint = np.zeros([len(self.X), len(self.Y)])
        
        for index, frec in self.frequencies.items():
            x = (index[0] - self.X[0])/self.step_X
            y = (index[1] - self.Y[0])/self.step_Y
            joint[int(x), int(y)] = frec
        
        return joint
    
    def get_Marginals(self, normalized=True):
        if normalized:
            marg_1 = self.joint_matrix.sum(axis=1)/self.N
            marg_2 = self.joint_matrix.sum(axis=0)/self.N
        else:
            marg_1 = self.joint_matrix.sum(axis=1)
            marg_2 = self.joint_matrix.sum(axis=0)
        return marg_1, marg_2
    
    def plot_joint_3d(self, joint_matrix = None, el=50, az=-5, ax=None, color='b', title=''):
        '''
        Grafica la probabilidad conjunta
        '''
        xpos, ypos = np.meshgrid(self.X, self.Y)
        xpos = xpos.T.flatten()
        ypos = ypos.T.flatten()
        
        zpos = np.zeros(xpos.shape)
        
        dx = self.step_X * np.ones_like(zpos)
        dy = self.step_Y * np.ones_like(zpos)
        
        if joint_matrix is None:
            dz = self.joint_matrix.astype(int).flatten()
        else:
            dz = joint_matrix.flatten()
        
        if ax == None:
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111, projection='3d')
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color, alpha=0.5)
        ax.set_title(title)
        ax.view_init(el, az)