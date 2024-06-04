import numpy as np

def get_out_strengths(G):
    out_strengths = {}
    for node in G.nodes():
        out_strengths[node] = 0
        for edge in G.out_edges(node, data=True):
            out_strengths[node] += edge[2]['weight']
    return out_strengths

def get_in_strengths(G):
    in_strengths = {}
    for node in G.nodes():
        in_strengths[node] = 0
        for edge in G.in_edges(node, data=True):
            in_strengths[node] += edge[2]['weight']
    return in_strengths

def get_D_inv(G):
    out_strengths = get_out_strengths(G)
    n = len(out_strengths)
    D_inv = np.zeros(shape=(n, n))
    for index, node in enumerate(G.nodes()):
        if out_strengths[node] == 0:
            D_inv[index][index] = 1
        else:
            D_inv[index][index] = 1 / out_strengths[node]
    return D_inv

def pageRank(A, Dinv, iters, G, q = 0.15):
    conv_history = []

    n = len(A)
    p = np.ones((n, 1)) / n
    out_str = get_out_strengths(G)
    in_str= get_in_strengths(G)

    deltaSj = np.zeros((n, 1))
    for i, node in enumerate(G.nodes()):
        if out_str[node] == 0:
            deltaSj[i] = 1

    for i in range(iters):
        pnew = (1-q) * np.matmul(np.matmul(A.transpose(), Dinv), p) + (q/n) + ((1-q)/n) * np.sum(np.multiply(p, deltaSj))
        change = np.sqrt(np.sum(np.power(p-pnew, 2)))
        conv_history.append(change)
        p = pnew / np.sum(pnew)
    
    return p, conv_history