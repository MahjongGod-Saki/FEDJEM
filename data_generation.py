import networkx as nx
import numpy as np
import scipy

##### simulation data #####


def create_adjacency_matrix(graph_num, graph_size, power_law_exponent):
    all_graphs = []
    for i in range(graph_num):
        g = nx.utils.powerlaw_sequence(graph_size, power_law_exponent)
        # power_law_exponent-->big[1.2, 5], node-->sparse.
        G = nx.expected_degree_graph(g, selfloops=False)
        all_graphs.append(nx.adjacency_matrix(G).todense())
    all_graphs = np.array(all_graphs)
    return all_graphs


def create_graphs(graph_num, graph_size, power_law_exponent, low, high, correction, times):
    all_graphs = create_adjacency_matrix(
        graph_num, graph_size, power_law_exponent)
    graph_result = []
    for graph in all_graphs:
        trans_matrix = np.random.uniform(low, high, (graph_size, graph_size))
        trans_matrix = np.where(
            trans_matrix > 0, trans_matrix + correction, trans_matrix - correction)
        graph = graph * trans_matrix
        vector = times * np.sum(abs(graph), axis=1)
        vector = np.where(vector == 0, 1, vector)
        graph = graph / vector[:, None]
        graph = 0.5 * (graph.transpose() + graph)
        vector2 = times * np.sum(abs(graph), axis=1) + 1
        graph = graph + np.diag(vector2)
        graph_result.append(graph)
    graph_result = np.array(graph_result)
    return all_graphs, graph_result


def inverse_block_diag_matrix(all_graphs):
    return np.linalg.inv(scipy.linalg.block_diag(*all_graphs))


def covariance_matrix(inv_A):
    if inv_A.shape[0] != inv_A.shape[1]:
        print('inv_A is not a square matrix.')
    else:
        A_size = inv_A.shape[0]
        diag_elements = abs(np.diagonal(inv_A))
        molecular = np.sqrt(np.outer(diag_elements, diag_elements))
        # 注意这里的 molecular没有考虑到可能为0的情况
        dij_matrix = 0.6 * np.ones((A_size, A_size)) + \
            0.4 * np.identity(A_size)
    return dij_matrix * inv_A / molecular


p = 128  # number of features
network_num = 3  # number of tasks
subnetwork_num = 6
shared_network_num = 4  # similarity
sparsity = 0.01  # range of sparsity[0, 1.0]
# print(sparsity)
exponent = 1.2 + sparsity * 4.8
n = 150 # num of observations
# step1: create ten shared graphs and record its adjacency matrix
adjacency_matrix, graphs = create_graphs(
    subnetwork_num, p, exponent, -0.3, 0.3, 0.1, 1.5)


# step2: three networks and its corresponding subnetwork
firstclass_graphs = graphs
secondclass_graphs = np.array([*graphs[:shared_network_num+1], np.identity(p)])
thirdclass_graphs = np.array(
    [*graphs[:shared_network_num], np.identity(p), np.identity(p)])


firstclass_cmatrix = covariance_matrix(
    inverse_block_diag_matrix(firstclass_graphs))
secondclass_cmatrix = covariance_matrix(
    inverse_block_diag_matrix(secondclass_graphs))
thirdclass_cmatrix = covariance_matrix(
    inverse_block_diag_matrix(thirdclass_graphs))


class1_adjacency_matrix = adjacency_matrix
class2_adjacency_matrix = np.array(
    [*adjacency_matrix[:shared_network_num+1], np.zeros((p, p))])
class3_adjacency_matrix = np.array(
    [*adjacency_matrix[:shared_network_num], np.zeros((p, p)), np.zeros((p, p))])
class1_graph = scipy.linalg.block_diag(*firstclass_graphs)
class2_graph = scipy.linalg.block_diag(*secondclass_graphs)
class3_graph = scipy.linalg.block_diag(*thirdclass_graphs)


class1_adjacency_matrix = scipy.linalg.block_diag(*class1_adjacency_matrix)
class2_adjacency_matrix = scipy.linalg.block_diag(*class2_adjacency_matrix)
class3_adjacency_matrix = scipy.linalg.block_diag(*class3_adjacency_matrix)

print(np.sum(class1_adjacency_matrix==1)+np.sum(class2_adjacency_matrix==1)+np.sum(class3_adjacency_matrix==1))

mean = np.zeros((p*subnetwork_num, ))
class1_simulation_data = np.random.multivariate_normal(
    mean, firstclass_cmatrix, n)
class2_simulation_data = np.random.multivariate_normal(
    mean, secondclass_cmatrix, n)
class3_simulation_data = np.random.multivariate_normal(
    mean, thirdclass_cmatrix, n)

