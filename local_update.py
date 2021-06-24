import numpy as np
import math
import multiprocessing as mp
import scipy
# import nilearn as nl


def update_Omega(rho, S, Z, U, n):
    W = S - rho * Z / n + rho * U / n
#     print('W', W)
    eigvalue, eigvector = np.linalg.eig(W)
    temp = np.copy(eigvalue)
#     print('eigvalue', eigvalue)
    for i in range(len(eigvalue)):
        temp[i] = n * (-eigvalue[i] + math.sqrt(abs(eigvalue[i] * eigvalue[i]) + 4 * rho / n)) / (2 * rho)
    D = np.diag(temp)
#     print('D', D)
    return eigvector @ D @ eigvector.T
