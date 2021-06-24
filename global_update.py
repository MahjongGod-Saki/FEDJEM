import pywt
import numpy as np


# global iter
def g_update_Z_step1(lambda1, rho, A_tensor):
    para = lambda1 / rho
    return pywt.threshold(A_tensor, para, mode='soft')


def g_update_Z_step2(soft_tensor, lambda2, rho):
    square_soft = np.square(soft_tensor)
    all_soft = np.sqrt(np.sum(square_soft, axis=0))
    all_soft = np.where(all_soft == 0, 1, all_soft)
    return np.maximum(1 - lambda2 / (rho * all_soft), 0)


def g_update_Z(lambda1, lambda2, rho, Omega_tensor, U_tensor):
    K = len(Omega_tensor)
    A_tensor = np.copy(Omega_tensor + U_tensor)
    update_Z_tensor = np.zeros(A_tensor.shape)
    soft_tensor = g_update_Z_step1(lambda1, rho, A_tensor)
    # print('soft_tensor', soft_tensor)
    all_soft = g_update_Z_step2(soft_tensor, lambda2, rho)
    # print('all_soft', all_soft)
    for k in range(K):
        # print('soft_tensor[k]', soft_tensor[k])
        # print('all_soft2', all_soft)
        update_Z_tensor[k] = soft_tensor[k] * all_soft
    return update_Z_tensor


# local iter
def update_U(U, Omega, Z):
    return U + Omega - Z
