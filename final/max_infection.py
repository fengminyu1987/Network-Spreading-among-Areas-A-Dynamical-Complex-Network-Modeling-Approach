# 最大感染人数
# 这个可以跟现实数据做叠图
# 将每个节点最大感染人数从大到小排列
# 09/26 200节点跑6min
# 新增：将距离感染发生点的距离叠图
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
import random
import json
import time
import pandas as pd

path = "images/max_infection/sir/"
H_vec = [1, 0.01, 1]
beta_gamma = [.5, .2]
s_i_r_init = [.9999, .0001, 0]
bottom = .2
top = .98
right = .98
left = .24
text_max_min = [-.005, -.03, 17]

n = 50
k = 4
c = .3

font_title = {'family': 'Times New Roman',
              'weight': 'bold',
              'style': 'italic',
              'size': 18,
              }
font_xy = {'family': 'Times New Roman',
           'weight': 'normal',
           'style': 'italic',
           'size': 30,
           }
font_axis = {'family': 'Times New Roman',
             'weight': 'normal',
             'style': 'italic',
             'size': 24,
             }


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


# 构建一个网络，返回值为网络内联矩阵
def build_network(n, k):
    p = .7
    g = nx.watts_strogatz_graph(n, k, p, seed=123)
    g = nx.barabasi_albert_graph(n, k, seed=123)
    fig_nw = plt.figure(figsize=figsize)
    nx.draw(g, pos=nx.spring_layout(g), node_size=10, edge_color="blue", node_color="red")

    with PdfPages('%snetwork.pdf' % path) as pdf:
        pdf.savefig(fig_nw)
    plt.savefig('%snetwork.eps' % path, format="eps")

    nx.write_graphml_lxml(g, "%sinfection.graphml" % path)

    inline_mat = [[0 for i in range(n)] for j in range(n)]

    for i in g.edges:
        strength = 1
        inline_mat[i[0]][i[1]] = strength
        inline_mat[i[1]][i[0]] = strength
    for i in range(len(inline_mat)):
        inline_mat[i][i] = -sum(inline_mat[i])
    # [print(i) for i in inline_mat]
    # [print("%d=%d" % (i, g.degree(i)), end=" ") for i in g.nodes]
    return (inline_mat, g)


def f1(x):
    if x % 3 == 0:
        return "S$_%d$" % (x / 3)
    elif x % 3 == 1:
        return "I$_%d$" % (x / 3)
    else:
        return "R$_%d$" % (x / 3)


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
    inline_mat, g = build_network(n, k)
    xs = [0 for i in range(3 * n)]
    paras = [0 for i in range(n * n)]
    for i in range(n):
        xs[i * 3] = s_i_r_init[0]
        xs[i * 3 + 1] = 0
        xs[i * 3 + 2] = 1 - s_i_r_init[0]
        for j in range(n):
            paras[i * n + j] = c * inline_mat[i][j]
    xs[1] = s_i_r_init[1]
    xs[2] = s_i_r_init[2]

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
    distance = [1 / len(i) for i in nx.single_source_shortest_path(g, 0).values()]
    # max_distance=max(distance)
    # distance=[max_distance-distance[i] for i in distance]
    d = {
        "infect": pd.Series(rate_infect, index=range(n)),
        "distance": pd.Series(distance, index=range(n)),
    }
    df = pd.DataFrame(d)
    df = df.sort_values(by='infect')
    # 绘制最大感染图与距离图
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(left=left, top=top, right=right)
    # plt.xlabel('N', fontdict=font_xy)
    plt.ylabel("max infected", fontdict=font_xy)
    plt.xticks(fontproperties=font_axis)
    plt.yticks(fontproperties=font_axis)
    plt.plot(range(n), sorted(rate_infect))

    plt.figure(figsize=figsize)
    plt.subplots_adjust(left=left, top=top, right=right)
    # plt.xlabel('N', fontdict=font_xy)
    plt.ylabel("distance", fontdict=font_xy)
    plt.xticks(fontproperties=font_axis)
    plt.yticks(fontproperties=font_axis)
    plt.plot(range(n), df["distance"])
    print(df)

    with PdfPages('%smax_infection.pdf' % path) as pdf:
        pdf.savefig(fig)
    plt.savefig('%smax_infection.eps' % path, format="eps")
    data = json.dumps([[list(z[i * 3 + j]) for j in range(3)] for i in range(n)])
    f = open("%sdata.json" % path, mode="w+")
    print(data, file=f)
    f.close()

    print(time.asctime())
    plt.show()
