import numpy as np
from sklearn.random_projection import SparseRandomProjection
def generarRP(n,m):
    density=int(np.sqrt(n))
    rp=SparseRandomProjection(n_components=m, density=density)
    rp.fit()
    