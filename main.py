from os import abort
import time
from tqdm import tqdm
import numpy as np
from multiprocessing.pool import ThreadPool
import scipy
import codecarbon

from local_update import update_Omega
from global_update import update_U, g_update_Z
from data_generation import p, class1_simulation_data, class2_simulation_data,\
    class3_simulation_data, network_num, subnetwork_num

from socket_rpc import init_rpc

pool = ThreadPool(5)
master = init_rpc("""
import math
import numpy as np
import time
from math import sqrt
import scipy
import codecarbon
S = None

@register
def set_worker_data(worker_data):
    global S
    S = worker_data.toarray()


tracker = codecarbon.EmissionsTracker()

@register
def update_Omega(rho, Z, U, n): 
    tracker.start()
    start_time = time.time()
    W = S - rho * Z.toarray() / n + rho * U.toarray() / n
    # print('W', W)
    eigvalue, eigvector = np.linalg.eig(W)
    temp = np.copy(eigvalue)
    # print('eigvalue', eigvalue)
    for i in range(len(eigvalue)):
        temp[i] = n * (-eigvalue[i] + sqrt(abs(eigvalue[i]
                       * eigvalue[i]) + 4 * rho / n)) / (2 * rho)
    D = np.diag(temp)
    # print('D', D)
    res = scipy.sparse.csr_matrix(eigvector @ np.diag(temp) @ eigvector.T)
    end_time = time.time()
    tracker.stop()
    return res, end_time - start_time

""")


def update_Omega_remote(thread_args):
    worker, args = thread_args
    start = time.time()
    result, local_time = master.rpc(worker, "update_Omega", *args)
    # result = master.rpc(worker, "update_Omega", *args)
    # communication_time = time.time() - start - local_time
    return result, local_time
    # return result


def complete_iteration(rho, lambda1, lambda2, X_list, Omega_tensor, U_tensor, Z_tensor, max_iter, penalty_choice):
    K = len(X_list)
    similar = []
    all_Omega = []
    all_Omega.append(np.copy(Omega_tensor))
    Sk = []
    nk = []
    local_time_list = []
    
    # communication_time_list = []
    # global_time_list = []
    for k in range(K):
        nk.append(X_list[k].shape[0])
        Sk.append(np.cov(X_list[k].T))

    bar = tqdm(range(max_iter), "iter")
    for i in bar:
        # start_time = time.time()

        # for k in range(K):
        #     Omega_tensor[k] = update_Omega(
        #         rho, Sk[k], Z_tensor[k], U_tensor[k], nk[k])  # 本地，多测几次
        # update_Omega_time = time.time() - start_time
        # communication_time = 0

        argslist = [(k, (rho, scipy.sparse.csr_matrix(Z_tensor[k]), scipy.sparse.csr_matrix(U_tensor[k]), nk[k]))
                    for k in range(K)]
        result = pool.map(update_Omega_remote, argslist)
        for k in range(K):
            Omega_tensor[k] = result[k][0].toarray()
        update_Omega_time = max([result[k][1] for k in range(K)])
        # communication_time = max([result[k][2] for k in range(K)])

        Z_tensor = g_update_Z(lambda1, lambda2, rho,
                              Omega_tensor, U_tensor)  # 服务器
        for k in range(K):
            U_tensor[k] = update_U(U_tensor[k], Omega_tensor[k], Z_tensor[k])
        all_Omega.append(np.copy(Omega_tensor))
        similar.append(Omega_similarity(all_Omega[i], all_Omega[i+1]))
        # global_time = time.time() - start_time - update_Omega_time - communication_time
        local_time_list.append(update_Omega_time)
        print(master.bytes_sent/(2**20))
        master.bytes_sent = 0
        print(master.bytes_recved/(2**20))
        master.bytes_recved = 0
        # communication_time_list.append(communication_time)
        # global_time_list.append(global_time)

        # bar.set_postfix(local_t=f"{','.join([f'{result[k][1]:.2f}' for k in range(K)])}",
        # comm_t=f"{communication_time:.2f}",
        # global_t=f"{global_time:.2f}")

    # all_time = np.array(
    #     [local_time_list, communication_time_list, global_time_list])
    # return Omega_tensor, Z_tensor, similar, all_time
    return Omega_tensor, Z_tensor, similar, local_time_list


def metric(result_matrix, original_matrix, threshold):
    if result_matrix.shape != original_matrix.shape:
        print("The shape of result_matrix or original matrix must be wrong!")
    result_matrix = np.where(abs(result_matrix) >= threshold, -1, 1)
    # np.savetxt('result_matrix.csv', result_matrix, delimiter=',')
    count_matrix = result_matrix - original_matrix
    TP = np.sum(count_matrix == 0)
    FP = np.sum(count_matrix == 1)
    TN = np.sum(count_matrix == -1)
    FN = np.sum(count_matrix == -2)
    confusion_matrix = [TP, FP, TN, FN]
    return confusion_matrix


def Omega_similarity(Omega_tensor1, Omega_tensor2):
    if Omega_tensor1.shape[0] != Omega_tensor2.shape[0]:
        print('The shape of Omega must be wrong!')
    similarity = 0.0
    # print('############')
    for i in range(Omega_tensor1.shape[0]):
        differ = Omega_tensor1[i] - Omega_tensor2[i]
        dist = np.linalg.norm(differ, ord='fro')
        denom = (np.linalg.norm(
            Omega_tensor1[i]) + np.linalg.norm(Omega_tensor2[i])) / 2
        similarity += 1 - (dist / denom)
    return similarity


# simulation data
if __name__ == '__main__':

    tracker = codecarbon.EmissionsTracker()
    tracker.start()

    # intialize the variables
    Omega0 = []
    U0 = []
    simulation_data = [class1_simulation_data,
                       class2_simulation_data, class3_simulation_data]

    for k in range(network_num):
        nk = simulation_data[k].shape[0]
        Sk = np.cov(simulation_data[k].T)
        master.rpc(k, "set_worker_data", scipy.sparse.csr_matrix(Sk))
        Omega0.append(np.diag(1/np.diag(Sk)))
        # Omega0.append(np.identity(p * subnetwork_num))
        U0.append(np.zeros((p * subnetwork_num, p * subnetwork_num)))
    Omega0 = np.array(Omega0)
    Z0 = np.copy(np.array(U0))
    Rho = 1e-3
    Lambda1 = 0.001
    Lambda2 = 0.03
    max_iter = 10
    # start_time = time.time()
    # Omega_result, Z_result, similarity, whole_time_list = complete_iteration(
    # Rho, Lambda1, Lambda2, simulation_data, Omega0, U0, Z0, max_iter, 'g')
    try:
        Omega_result, Z_result, similarity, local_time = complete_iteration(
            Rho, Lambda1, Lambda2, simulation_data, Omega0, U0, Z0, max_iter, 'f')
    except KeyboardInterrupt:
        pass
    print(local_time)
    print(np.sum(local_time))
    # end_time = time.time()
    # print(similarity)
    # print(whole_time_list)

    tracker.stop()

    # print(f"time: {end_time - start_time:.3f}s")
    abort()
