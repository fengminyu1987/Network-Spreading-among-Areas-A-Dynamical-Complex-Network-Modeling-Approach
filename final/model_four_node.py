# 画出模型在少量节点时的表现
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
import random
import json
import time
import pandas as pd
import os

network_kind = "hah"
network_kinds = ["BA", "WS", "regular_random", "NW", "ER", "regular"]
seed = 123

path = "images/four_nodes/%s/" % network_kind
if not os.path.exists(path):
    os.mkdir(path)

paras = json.load(open("parameters.json"))
print(paras)

H_vec = paras["H"]
beta_gamma = paras["beta_gamma"]
s_i_r_init = [1, .01, 0]
bottom = .02
top = .9
right = .98
left = .02
text_max_min = [-.005, -.03, 17]
p = paras["p"]

n = 4
k = paras["k"]
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


# 只是没删那些有的没的 实际上只有四个节点
def get_network(n, k, p=1):
    g = nx.Graph()
    N = [0, 1, 2, 3]
    g.add_nodes_from(N)
    g.add_edge(0, 1)

    g.add_edge(0, 2)
    #g.add_edge(1, 3)

    g.add_edge(0, 3)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    return g


# 构建一个网络，返回值为网络内联矩阵
def build_network(n, k):
    g = get_network(n, k, p)
    # fig_nw = plt.figure(figsize=figsize)
    # nx.draw(g, pos=nx.spring_layout(g), node_size=10, edge_color="blue", node_color="red")
    #
    # with PdfPages('%snetwork.pdf' % path) as pdf:
    #     pdf.savefig(fig_nw)
    # plt.savefig('%snetwork.eps' % path, format="eps")

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


# def f1(x):
#     if x % 3 == 0:
#         return "S$_%d$" % (x / 3)
#     elif x % 3 == 1:
#         return "I$_%d$" % (x / 3)
#     else:
#         return "R$_%d$" % (x / 3)


# 返回某节点感染的比例
def rate_infection(z):
    result = []
    mean_result = max([np.mean([z[i * 3 + 1][j] / (z[i * 3][j] + z[i * 3 + 1][j] + z[i * 3 + 2][j])
                                for i in range(n)]) for j in range(len(z[1]))])
    for i in range(n):
        result.append(max([z[i * 3 + 1][j] / (z[i * 3][j] + z[i * 3 + 1][j] + z[i * 3 + 2][j])
                           for j in range(len(z[i * 3 + 1]))]))
    return result, mean_result


def synchronized_time():
    inline_mat, g = build_network(n, k)

    length = 100

    start_p = 0
    synchronization_upsilon = .001

    xs = [0 for i in range(3 * n)]
    paras = [0 for i in range(n * n)]
    for i in range(n):
        xs[i * 3] = s_i_r_init[0]
        xs[i * 3 + 1] = 0
        xs[i * 3 + 2] = 1 - s_i_r_init[0]
        for j in range(n):
            paras[i * n + j] = -c * inline_mat[i][j]
    xs[start_p * 3 + 1] = s_i_r_init[1]
    xs[start_p * 3] = 1 - s_i_r_init[1]
    print(xs[start_p * 3])

    print(time.asctime())
    sol = solve_ivp(sir, [0, length], xs, args=paras, method='Radau',
                    dense_output=True)
    t = np.linspace(0, length, 30)
    z = sol.sol(t)
    print(time.asctime())

    aver_x = [[0 for _ in range(len(t))] for _ in range(3)]
    synchronized_t = [[0 for _ in range(n)] for _ in range(3)]

    for j in range(3):
        for i in range(n):
            for h in range(len(t)):
                aver_x[j][h] += z.T.T[i * 3 + j][h]
        aver_x[j] = [i / n for i in aver_x[j]]
        for i in range(n):
            is_synchronized = False
            for ka in range(1, len(t)):
                if abs(z[i * 3 + j][ka] - aver_x[j][ka]) > synchronization_upsilon:
                    is_synchronized = False
                elif not is_synchronized and ka > synchronized_t[j][i]:
                    synchronized_t[j][i] = t[ka]
                    is_synchronized = True
    final_time = int(max([max(i) for i in synchronized_t]))
    # final_time=30
    print(final_time)

    figsize = (4, 3)

    # ts = [0, int(final_time / 4 / length * len(t)), int(final_time / 2 / length * len(t)),
    #       int(final_time / length * len(t))]
    figs_n = 8
    ts = [i * int(final_time / figs_n / length * len(t)) for i in range(figs_n+1)]

    for i in ts:
        temp = [[0 for _ in range(n)] for _ in range(3)]
        max_infected = max([max(z.T.T[h * 3 + 1]) for h in range(n)])
        for h in range(n):
            sum_t = z.T.T[h * 3][i] + z.T.T[h * 3 + 1][i] + z.T.T[h * 3 + 2][i]
            for j in range(3):
                temp[j][h] = z.T.T[h * 3 + j][i] / sum_t
        fig_nw = plt.figure(figsize=figsize)
        plt.subplots_adjust(top=top, right=right, bottom=bottom, left=left)
        plt.title("t=%d" % int(length * i / len(t)),fontdict=font_title)
        g2 = nx.Graph(g)
        for j in range(4):
            g2.add_edge(j, 4 + j * 3)
            g2.add_edge(j, 5 + j * 3)
            g2.add_edge(j, 6 + j * 3)

        pos = nx.kamada_kawai_layout(g2)
        options = {"edgecolors": "tab:gray", "node_size": 800}
        for j in range(4):
            nx.draw_networkx_nodes(g2, pos, nodelist=[j], node_color=(temp[1][j] / max_infected if temp[1][
                                                                                                       j] / max_infected < 1 else 1,
                                                                      temp[2][j] if temp[2][j] > .5 else 0,
                                                                      temp[0][j] if temp[0][j] > .5 else 0), **options)
            nx.draw_networkx_nodes(g2, pos, nodelist=[j], node_color="white", **{"edgecolors": "tab:gray", "node_size": 500})
        # nx.draw_networkx_nodes(g2, pos, nodelist=range(4, 16),node_color="white",**options)
        labels = {}
        labels2 = {0: {}, 1: {}, 2: {}}

        # edges
        nx.draw_networkx_edges(g, pos, width=6.0, alpha=0.5)
        nx.draw_networkx_edges(g2, pos, width=4.0, alpha=0.3)

        bbox = {
                'facecolor': '#F0FFFF', #填充色
                'alpha': .7,  # 框透明度

                }

        for j in range(4):
            labels[j] = j
            labels2[0][4 + j * 3] = "S:%.2f" % temp[0][j]
            labels2[1][5 + j * 3] = "I:%.2f" % temp[1][j]
            labels2[2][6 + j * 3] = "R:%.2f" % temp[2][j]
        nx.draw_networkx_labels(g, pos, labels, font_size=22, font_color="black")
        nx.draw_networkx_labels(g2, pos, labels2[0], font_size=11, font_color="blue",bbox=bbox)
        nx.draw_networkx_labels(g2, pos, labels2[1], font_size=11, font_color="red",bbox=bbox)
        nx.draw_networkx_labels(g2, pos, labels2[2], font_size=11, font_color="green",bbox=bbox)
        with PdfPages('%snetwork%d.pdf' % (path, int(length * i / len(t)))) as pdf:
            pdf.savefig(fig_nw)
        plt.savefig('%snetwork%d.eps' % (path, int(length * i / len(t))), format="eps")


if __name__ == '__main__':
    # f = open("%sdata.json" % path, mode="w+")
    # print(data, file=f)
    # f.close()
    synchronized_time()

    plt.show()
