# 最大感染人数与实际数据叠图

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
H_vec = [1, 0.015, 1]
beta_gamma = [.255, .18]
s_i_r_init = [.999, .001, 0]
bottom = .25
top = .98
right = .98
left = .18
text_max_min = [-.005, -.03, 17]

n = 35
k = 10
# c = .085
c = .2

font_title = {'family': 'Times New Roman',
              'weight': 'bold',
              'style': 'italic',
              'size': 18,
              }
font_xy = {'family': 'Times New Roman',
           'weight': 'normal',
           'style': 'italic',
           'size': 24,
           }
font_axis = {'family': 'Times New Roman',
             'weight': 'normal',
             'style': 'italic',
             'size': 16,
             }

figsize = (6, 2.5)

def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(np.array(t)*np.log(np.array(y)+delta))

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
    g = nx.watts_strogatz_graph(n, k, p, seed=123)
    # g = nx.barabasi_albert_graph(n, k, seed=123)
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


def max_infection_in():
    data = pd.read_csv("covid_19_data.csv")
    data = data[data["ObservationDate"] < "07/01/2021"]
    data = data[data["Country/Region"] == "India"]
    data = data.drop(index=data[data['Province/State'] == "Unknown"].index)
    
    population_data = pd.read_csv("POPULATION_IN.csv")
    province = population_data["Province/State"].values
    print(province)
    population_data = population_data.set_index("Province/State")
    # print(population_data)

    max_infect_list = []
    max_infect_distribution = []
    
    for i in province:
        data_temp = data[data["Province/State"] == i]
        population_province = population_data.at[i, "Population"]
        if len(data_temp["Confirmed"].values)==0:
            continue
        max_infect_list.append(max(data_temp["Confirmed"].values) / population_province)
    print("max infect list ",max_infect_list)
    return sorted(max_infect_list)
    # plt.hist(sorted(max_infect_list), 20, density=1, facecolor='g', alpha=0.75)
    # plt.bar(range(len(max_infect_list)), sorted(max_infect_list), alpha=0.5, color="grey")


# 返回某节点感染的比例
def rate_infection(z):
    result = []
    mean_result = max([np.mean([z[i * 3 + 1][j] / (z[i * 3][j] + z[i * 3 + 1][j] + z[i * 3 + 2][j])
                                for i in range(n)]) for j in range(len(z[1]))])
    for i in range(n):
        result.append(max([z[i * 3 + 1][j] / (z[i * 3][j] + z[i * 3 + 1][j] + z[i * 3 + 2][j])
                           for j in range(len(z[i * 3 + 1]))]))
    return result, mean_result


print(time.asctime())
length = 100
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

# %%

def got_distribution(list, n_column, max, min,interval):
    temp = [0 for _ in range(n_column)]
    for i in range(n_column):
        for j in list:
            if j < min + (i+1) * interval:
                temp[i] += 1
    return [i/len(list) for i in temp]

rate_infect, mean_infect = rate_infection(z)
print("simulation average ", np.mean(rate_infect))
print("simulation max ", np.max(rate_infect))
print("simulation cv ", np.std(rate_infect) / np.mean(rate_infect))
distance = [1 / len(i) for i in nx.single_source_shortest_path(g, 0).values()]
# max_distance=max(distance)
# distance=[max_distance-distance[i] for i in distance]
d = {
    "infect": pd.Series(rate_infect, index=range(n)),
    "distance": pd.Series(distance, index=range(n)),
}
df = pd.DataFrame(d)
df = df.sort_values(by='infect')

max_infect_list = max_infection_in()
print(len(max_infect_list))

print("IN average ", np.mean(max_infect_list))
print("IN max ", np.max(max_infect_list))
print("IN cv ", np.std(max_infect_list) / np.mean(max_infect_list))

cols=100
interval=.01

max_infect=max(max(rate_infect),max(max_infect_list))
for i in range(cols):
    if i*interval>max_infect:
        max_infect=cols*interval
        cols=i
        break

# 绘制最大感染图
fig = plt.figure(figsize=figsize)
plt.subplots_adjust(left=left, top=top, right=right,bottom=.16)
plt.xlabel('States', fontdict=font_xy)
plt.ylabel("PIR", fontdict=font_xy)
plt.yticks([i*interval for i in range(0,cols+1,2)],
           ["%.1f%%"%(100*i*interval) for i in range(0,cols+1,2)]
    ,fontproperties=font_axis)
plt.plot(range(n), sorted(rate_infect),color="orange",label="simulation")
plt.fill_between(range(n), sorted(rate_infect),
                 alpha=.3,color="orange")

plt.bar(range(len(max_infect_list)), max_infect_list, 
        alpha=.9,color="violet",label="IN")
plt.xticks([])
plt.legend()

with PdfPages('%smax_infection_IN.pdf' % path) as pdf:
    pdf.savefig(fig)
plt.savefig('%smax_infection_IN.eps' % path, format="eps")
data = json.dumps([[list(z[i * 3 + j]) for j in range(3)] for i in range(n)])
f = open("%sdata.json" % path, mode="w+")
print(data, file=f)
f.close()

# %%
fig = plt.figure(figsize=figsize)
plt.subplots_adjust(left=left, top=top, right=right,bottom=bottom)


plt.plot(got_distribution
         (max_infect_list,cols,
          max_infect, 0,interval=interval),"o-",markerfacecolor="w",label="IN")
plt.plot(got_distribution(rate_infect,
                          cols,
                          max_infect, 0,interval=interval),label="simulation")

print("cross entropy: ",cross_entropy_error(got_distribution
         (max_infect_list,cols,
          max_infect, 0,interval=interval),
                        got_distribution(rate_infect,
                          cols,
                          max_infect, 0,interval=interval)))

plt.fill_between(range(0,cols), got_distribution
         (max_infect_list,cols,
          max_infect, 0,interval=interval), 
         got_distribution(rate_infect,
                cols,max_infect, 0,interval=interval),
         color="silver",hatch="|",alpha=.5)

plt.xticks(range(0,cols,2),
           ["%.2f%%"%(i*interval*100) for i in range(0,cols,2)],
           fontproperties=font_axis,)
plt.yticks(fontproperties=font_axis)
plt.ylabel("p", fontdict=font_xy)
plt.xlabel("PIR", fontdict=font_xy)
plt.legend()
with PdfPages('%sdistribution_IN.pdf' % path) as pdf:
    pdf.savefig(fig)
plt.savefig('%sdistribution_IN.eps' % path, format="eps")
print(time.asctime())
plt.show()
