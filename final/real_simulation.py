import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
import random
import json
import time
import pandas as pd

path = "./"
H_vec = [1, 0.025, 1]
beta_gamma = [.26, .18]
s_i_r_init = [.999, .001, 0]
bottom = .25
top = .98
right = .98
left = .18
figsize = (6, 2.5)
nation_dict = {"US": [52, .31, 10], "CN": [31, .185, 4], "IN": [35, .265, 5]}
c = .2

path = "images/real_simulation/"

# %%

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
    p = .5
    g = nx.watts_strogatz_graph(n, k, p)
    # g = nx.barabasi_albert_graph(n, k)
    # fig_nw = plt.figure(figsize=figsize)
    # nx.draw(g, pos=nx.spring_layout(g), node_size=10, edge_color="blue", node_color="red")
    #
    # with PdfPages('%snetwork.pdf' % path) as pdf:
    #     pdf.savefig(fig_nw)
    # plt.savefig('%snetwork.eps' % path, format="eps")
    #
    # nx.write_graphml_lxml(g, "%sinfection.graphml" % path)
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


def max_infection_us():
    data = pd.read_csv("covid_19_data.csv")
    data = data[data["ObservationDate"] < "07/01/2021"]
    data = data[data["Country/Region"] == "US"]
    data = data.drop(index=data[data['Province/State'] == "Unknown"].index)

    population_data = pd.read_csv("POPULATION_US.csv")
    province = population_data["Province/State"].values

    population_data = population_data.set_index("Province/State")
    # print(population_data)

    max_infect_list = []
    max_infect_distribution = []

    for i in province:
        data_temp = data[data["Province/State"] == i]
        population_province = population_data.at[i, "Population"]
        max_infect_list.append(max(data_temp["Confirmed"].values) / population_province)
    print(max_infect_list)
    return sorted(max_infect_list)
    # plt.hist(sorted(max_infect_list), 20, density=1, facecolor='g', alpha=0.75)
    # plt.bar(range(len(max_infect_list)), sorted(max_infect_list), alpha=0.5, color="grey")


# 返回某节点感染的比例
def rate_infection(z, n):
    result = []
    mean_result = max([np.mean([z[i * 3 + 1][j] / (z[i * 3][j] + z[i * 3 + 1][j] + z[i * 3 + 2][j])
                                for i in range(n)]) for j in range(len(z[1]))])
    for i in range(n):
        result.append(max([z[i * 3 + 1][j] / (z[i * 3][j] + z[i * 3 + 1][j] + z[i * 3 + 2][j])
                           for j in range(len(z[i * 3 + 1]))]))
    return result, mean_result


print(time.asctime())
length = 100

data = {}

for country in nation_dict.keys():
    data[country] = []
    n = nation_dict[country][0]
    beta_gamma[0] = nation_dict[country][1]
    k = nation_dict[country][2]
    for batch in range(10):
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
        rate_infect, mean_infect = rate_infection(z, n)
        data[country].append(rate_infect)
data = json.dumps(data)
f = open("%sdata.json" % path, mode="w+")
print(data, file=f)
f.close()