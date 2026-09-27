# 最大节点感染比例与平均最大感染比例
# 根据不同beta画出多线条
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
import os
import json
import time
import pandas as pd

network_kind = "BA"
network_kinds = ["BA", "WS", "regular_random", "NW", "ER", "regular"]
seed = 123

paras = json.load(open("parameters.json"))

k = 8

path = "images/infection_ratio_strength_line/%s%d/" % (network_kind,k)
if not os.path.exists(path):
    os.mkdir(path)


H_vec = paras["H"]
beta_gamma = paras["beta_gamma"]
s_i_r_init = [1, .001, 0]
bottom = .2
top = .98
right = .98
left = .24
text_max_min = [-.005, -.03, 17]
p = paras["p"]

n = 100

c = paras["c"]

font_title = paras["font_title"]
font_xy = paras["font_xy"]
font_axis = paras["font_axis"]

def sir(t, z, *args):
    result = []
    kind_len = int(len(list(z)) / 3)
    for j in range(kind_len):
        s = list(z)[j * 3]
        i = list(z)[j * 3 + 1]
        beta = beta_gamma[0]
        gamma = beta_gamma[1]
        # 孤立节点方程
        s0 = - beta * s * i
        i0 = beta * s * i - gamma * i
        r0 = gamma * i
        # 耦合项
        for h in range(j * kind_len, (j + 1) * kind_len):
            sj = z[3 * (h - (j * kind_len))]
            ij = z[3 * (h - (j * kind_len)) + 1]
            rj = z[3 * (h - (j * kind_len)) + 2]
            s0 += H_vec[0] * sj * args[h]
            i0 += H_vec[1] * ij * args[h]
            r0 += H_vec[2] * rj * args[h]
        result.append(s0)
        result.append(i0)
        result.append(r0)
    return result


def get_network(n, k, p=1):
    if network_kind == network_kinds[0]:
        return nx.barabasi_albert_graph(n, k, seed=seed)
    elif network_kind == network_kinds[1]:
        return nx.watts_strogatz_graph(n, k, p, seed=seed)
    elif network_kind == network_kinds[2]:
        return nx.random_regular_graph(k, n)
    elif network_kind == network_kinds[3]:
        return nx.newman_watts_strogatz_graph(n, k, p, seed=seed)
    elif network_kind == network_kinds[4]:
        return nx.erdos_renyi_graph(n, p)
    elif network_kind == network_kinds[5]:
        g = nx.Graph()
        N = list(range(n))
        g.add_nodes_from(N)
        for i in range(n):
            for j in range(1, int(k / 2) + 1):
                g.add_edge(i, (i + j) % n)
        return g


# 构建一个网络，返回值为网络内联矩阵
def build_network(n, k):
    g = get_network(n, k, p)
    fig_nw = plt.figure(figsize=figsize)
    nx.draw(g, pos=nx.spring_layout(g), node_size=10, edge_color="blue", node_color="red")

    with PdfPages('%snetwork.pdf' % path) as pdf:
        pdf.savefig(fig_nw)
    plt.savefig('%snetwork.eps' % path, format="eps")

    nx.write_graphml_lxml(g, "%sinfection.graphml" % path)

    inline_mat = [[0 for i in range(n)] for j in range(n)]

    for i in g.edges:
        strength = -1
        inline_mat[i[0]][i[1]] = strength
        inline_mat[i[1]][i[0]] = strength
    for i in range(len(inline_mat)):
        inline_mat[i][i] = -sum(inline_mat[i])
    # [print(i) for i in inline_mat]
    # [print("%d=%d" % (i, g.degree(i)), end=" ") for i in g.nodes]
    return (inline_mat, g)

def got_distribution(list, n_column, max, min):
    temp = [0 for _ in range(n_column)]
    for i in range(n_column):
        for j in list:
            if j < min + (i+1) * (max - min) / n_column:
                    temp[i] += 1
    return temp


# 返回某节点感染的比例
def rate_infection(z):
    result = []
    mean_result = max([np.mean([z[i * 3 + 1][j] / (z[i * 3][j] + z[i * 3 + 1][j] + z[i * 3 + 2][j])
                                for i in range(n)]) for j in range(len(z[1]))])
    for i in range(n):
        result.append(max([z[i * 3 + 1][j] / (z[i * 3][j] + z[i * 3 + 1][j] + z[i * 3 + 2][j])
                           for j in range(len(z[i * 3 + 1]))]))
    return result, mean_result


if __name__ == '__main__':
    print(time.asctime())
    length = 100
    figsize = (5, 4.2)
    begin = 20
    end = 60
    step = 15
    rate_describe_dict = {}
    strengths = [i / 100 for i in range(begin, end, step)]
    inline_mat, g = build_network(n, k)

    ps = sorted(g.degree(g.nodes()), key=lambda x: (x[1]))
    start_p = ps[int(len(ps) / 2)][0]

    # 绘制最大感染图
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(left=left, top=top, right=right, bottom=bottom)
    plt.xlabel('beta', fontdict=font_xy)
    plt.ylabel("max infected", fontdict=font_xy)
    plt.xticks(fontproperties=font_axis, rotation=30)
    plt.yticks(fontproperties=font_axis)

    for strength in strengths:
        #beta_gamma[0]=beta
        print("strength= ", strength, " ", time.asctime())
        xs = [0 for i in range(3 * n)]
        paras = [0 for i in range(n * n)]
        for i in range(n):
            xs[i * 3] = s_i_r_init[0]
            xs[i * 3 + 1] = 0
            xs[i * 3 + 2] = 1 - s_i_r_init[0]
            for j in range(n):
                paras[i * n + j] = -strength * inline_mat[i][j]
        xs[start_p * 3 + 1] = s_i_r_init[1]
        xs[start_p * 3] = 1 - s_i_r_init[1]

        sol = solve_ivp(sir, [0, length], xs, args=paras, method='Radau',
                        dense_output=True)
        t = np.linspace(0, length, 30)
        z = sol.sol(t)
        sol_isolate = solve_ivp(sir, [0, length], [(xs[0] + 1 * n - 1) / n, xs[1] / n, xs[2] / n],
                                args=[paras[0], paras[1], 0],
                                method='Radau',
                                dense_output=True)
        z_isolate = sol_isolate.sol(t)

        rate_infect, mean_infect = rate_infection(z)

        plt.plot(range(n), sorted(rate_infect),label=strength)

        rate_describe_dict[strength] = rate_infect

    plt.legend()
    with PdfPages('%sinfection.pdf' % path) as pdf:
        pdf.savefig(fig)
    plt.savefig('%sinfection.eps' % path, format="eps")

    data = json.dumps({"infection_ratio": rate_describe_dict,
                       "network_kind": network_kind,
                       "n": n,
                       "k": k,
                       "p": p,
                       "variable": "strength"})

    f = open("%sdata.json" % path, mode="w+")
    print(data, file=f)
    f.close()

    print(time.asctime())
    plt.show()
