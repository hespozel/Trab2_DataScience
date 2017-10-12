from __future__ import division
from sklearn import manifold
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


os.chdir("C:/Users/hespo/OneDrive/Documentos/GitHub/Trab2_DataScience")
cwd = os.getcwd()
print (cwd)

pd.options.display.max_columns = 999
pd.options.display.max_rows = 999


# Distance file available from RMDS project:
#    https://github.com/cheind/rmds/blob/master/examples/european_city_distances.csv
reader = csv.reader(open("city_distances.csv", "r"), delimiter=';')
data = list(reader)

def cmdscale(D):
    """
    Classical multidimensional scaling (MDS)

    Parameters
    ----------
    D : (n, n) array
        Symmetric distance matrix.

    Returns
    -------
    Y : (n, p) array
        Configuration matrix. Each column represents a dimension. Only the
        p dimensions corresponding to positive eigenvalues of B are returned.
        Note that each dimension is only determined up to an overall sign,
        corresponding to a reflection.

    e : (n,) array
        Eigenvalues of B.

    """
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # YY^T
    B = -H.dot(D ** 2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)

    return Y, evals


dists = []
cities = []
for d in data:
    cities.append(d[0])
    dists.append(map(float , d[1:]))

adist = np.array(dists)
print (adist.shape)
amax = np.amax(adist)
adist /= amax
print (adist.shape)

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
results = mds.fit(adist)
results = results
coords = results.embedding_
print (coords)
#coords = 1000 * np.array(results.embedding_)
plt.subplots_adjust(bottom = 0.1)
plt.scatter(
    coords[:, 1], coords[:, 0],  marker = 'o'
    )
for label, x, y in zip(cities, coords[:, 1], coords[:, 0]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()