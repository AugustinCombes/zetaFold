import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def normalize_adjacency(A):
    """
    Function that normalizes an adjacency matrix
    """
    n = A.shape[0]
    A += sp.identity(n)
    degs = A.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D = sp.diags(inv_degs)
    A_normalized = D.dot(A)

    return A_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Function that converts a Scipy sparse matrix to a sparse Torch tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(): 
    """
    Function that loads graphs
    """  
    graph_indicator = np.loadtxt("data/graph_indicator.txt", dtype=np.int64)
    _,graph_size = np.unique(graph_indicator, return_counts=True)
    
    edges = np.loadtxt("data/edgelist.txt", dtype=np.int64, delimiter=",")
    edges_inv = np.vstack((edges[:,1], edges[:,0]))
    edges = np.vstack((edges, edges_inv.T))
    s = edges[:,0]*graph_indicator.size + edges[:,1]
    idx_sort = np.argsort(s)
    edges = edges[idx_sort,:]
    edges,idx_unique =  np.unique(edges, axis=0, return_index=True)
    A = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))
    
    x = np.loadtxt("data/node_attributes.txt", delimiter=",")
    edge_attr = np.loadtxt("data/edge_attributes.txt", delimiter=",")
    edge_attr = np.vstack((edge_attr,edge_attr))
    edge_attr = edge_attr[idx_sort,:]
    edge_attr = edge_attr[idx_unique,:]
    
    adj = []
    features = []
    edge_features = []
    idx_n = 0
    idx_m = 0
    for i in range(graph_size.size):
        adj.append(A[idx_n:idx_n+graph_size[i],idx_n:idx_n+graph_size[i]])
        edge_features.append(edge_attr[idx_m:idx_m+adj[i].nnz,:])
        features.append(x[idx_n:idx_n+graph_size[i],:])
        idx_n += graph_size[i]
        idx_m += adj[i].nnz

    return adj, features, edge_features

def load_data(): 
    """
    Function that loads graphs
    """  
    graph_indicator = np.loadtxt("data/graph_indicator.txt", dtype=np.int64)
    _,graph_size = np.unique(graph_indicator, return_counts=True)
    x = np.loadtxt("data/node_attributes.txt", delimiter=",")
    edge_attr = np.loadtxt("data/edge_attributes.txt", delimiter=",")
    edges = np.loadtxt("data/edgelist.txt", dtype=np.int64, delimiter=",")
    
    A = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))
    A.setdiag(0)
    A += A.T

    W = sp.csr_matrix((1/edge_attr[:, 0], (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size)) # edge weights
    W.setdiag(0)
    W += W.T
    
    adj = []
    adj_weight = []
    features = []
    edge_features = []
    idx_n = 0
    idx_m = 0
    for i in range(graph_size.size):
        adj.append(A[idx_n:idx_n+graph_size[i],idx_n:idx_n+graph_size[i]])
        adj_weight.append(W[idx_n:idx_n+graph_size[i],idx_n:idx_n+graph_size[i]])
        edge_features.append(edge_attr[idx_m:idx_m+adj[i].nnz,:])
        features.append(x[idx_n:idx_n+graph_size[i],:])
        idx_n += graph_size[i]
        idx_m += adj[i].nnz

    return adj, adj_weight, features, edge_features


def normalize_adjacency(A, W):
    """
    Function that normalizes an adjacency matrix and includes edge weights
    """
    n = A.shape[0]
    degs = A.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D_inv = sp.diags(inv_degs)
    L_rw = sp.identity(n) - D_inv.dot(W)

    return L_rw