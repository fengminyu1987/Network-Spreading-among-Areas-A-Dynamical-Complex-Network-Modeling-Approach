# 画出横坐标为节点数目 纵坐标为同步时间的图（可能有点用）
# 同时画出横坐标为节点数目 纵坐标为同步状态的图（并没有用）
# 09/08 新增：从未受感染的区域数目（并没有用）
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
import random

path = "images/time_n/sir/"
n = 16
k = 4
synchronization_upsilon = .001
length = 100
n_length = 10
figsize = (5, 3.5)
c = .2

H_vec = [1, 0.01, 1]
beta_gamma = [.5, .2]
s_i_r_init = [.9999, .0001, 0]
bottom = .2
top = .98
right = .98
left = .15
text_max_min = [-.005, -.03, 17]

font_title = {'family': 'Times New Roman',
              'weight': 'bold',
              'style': 'italic',
              'size': 18,
              }
font_xy = {'family': 'Times New Roman',
           'weight': 'normal',
           'style': 'italic',
           'size': 18,
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
    inline_mat = [[0 for i in range(n)] for j in range(n)]

    for i in g.edges:
        strength = 1
        inline_mat[i[0]][i[1]] = strength
        inline_mat[i[1]][i[0]] = strength
    for i in range(len(inline_mat)):
        inline_mat[i][i] = -sum(inline_mat[i])
    # [print(i) for i in inline_mat]
    # [print("%d=%d" % (i, g.degree(i)), end=" ") for i in g.nodes]
    return inline_mat


def f1(x):
    if x % 3 == 0:
        return "S$_%d$" % (x / 3)
    elif x % 3 == 1:
        return "I$_%d$" % (x / 3)
    else:
        return "R$_%d$" % (x / 3)


def synchronized_time_fun(t, z, aver_x,n):
    synchronized_t = [0, 0, 0]
    for j in range(3):
        for i in range(n):
            for h in range(len(t)):
                aver_x[j][h] += z.T.T[i * 3 + j][h]
        aver_x[j] = [i / n for i in aver_x[j]]
        for i in range(n):
            is_synchronized = False
            for time in range(1, len(t)):
                if abs(z[i * 3 + j][time] - aver_x[j][time]) / aver_x[j][time] > synchronization_upsilon:
                    is_synchronized = False
                elif not is_synchronized and time > synchronized_t[j]:
                    synchronized_t[j] = t[time]
                    is_synchronized = True
    return synchronized_t


def no_infection(t, z, n):
    number = 0
    for i in range(n):
        zero_infection = True
        for time in range(len(t)):
            if z[i * 3 + 1][time] > 0:
                zero_infection = False
                break
        if zero_infection:
            number += 1
    return number


def run_strength_time(n):
    inline_mat = build_network(n, k)
    xs = [0 for _ in range(3 * n)]
    paras = [0 for _ in range(n * n)]
    for i in range(n):
        xs[i * 3] = s_i_r_init[0]
        xs[i * 3 + 1] = 0
        xs[i * 3 + 2] = 1 - s_i_r_init[0]
        for j in range(n):
            paras[i * n + j] = c * inline_mat[i][j]
    xs[0] = 1 - s_i_r_init[1]
    xs[1] = s_i_r_init[1]
    xs[2] = s_i_r_init[2]

    sol = solve_ivp(sir, [0, length], xs, args=paras, method='Radau',
                    dense_output=True)
    t = np.linspace(0, length, length * n_length)
    z = sol.sol(t)
    aver_x = [[0 for _ in range(len(t))] for _ in range(3)]

    synchronized_t = synchronized_time_fun(t, z, aver_x,n)

    return synchronized_t, [aver_x[0][-1], aver_x[1][-1], aver_x[2][-1]], no_infection(t, z, n)


if __name__ == '__main__':
    n0 = 2
    synchronized_c_time = [[], [], []]
    synchronized_c_state = [[], [], []]
    synchronized_c_no_infect = []
    n = range(9, 29)
    for i in n:
        temp1, temp2, temp3 = run_strength_time(i * n0 + n0)
        for j in range(3):
            synchronized_c_time[j].append(temp1[j])
            synchronized_c_state[j].append(temp2[j])
        synchronized_c_no_infect.append(temp3)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=top, right=right, bottom=bottom, left=left)
    for j in range(3):
        plt.plot([i * n0 + n0 for i in n], synchronized_c_time[j], "o-",
                 label=["S", "I", "R"][j], markerfacecolor="w")
    plt.xticks([i * n0 + n0 for i in n[::2]], rotation=30)
    plt.yticks([i * 20 for i in range(1, 1 + int(length / 20))])
    plt.xlabel("N", fontdict=font_xy)
    plt.ylabel("time", fontdict=font_xy)
    plt.legend()
    with PdfPages('%sn_time.pdf' % path) as pdf:
        pdf.savefig(fig)
    plt.savefig('%sn_time.eps' % path, format="eps")

    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=top, right=right, bottom=bottom, left=left)
    for j in range(3):
        plt.plot([i * n0 + n0 for i in n], synchronized_c_state[j], "o-",
                 label=["S", "I", "R"][j], markerfacecolor="w")
    # plt.xticks([i * c0 + c0 for i in c[::2]], rotation=30)
    # plt.yticks([i * 20 for i in range(1, 1 + int(length / 20))])
    plt.xlabel("N", fontdict=font_xy)
    plt.ylabel("state", fontdict=font_xy)
    plt.legend()
    with PdfPages('%sn_state.pdf' % path) as pdf:
        pdf.savefig(fig)
    plt.savefig('%sn_state.eps' % path, format="eps")

    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=top, right=right, bottom=bottom, left=left)
    plt.plot([i * n0 + n0 for i in n], synchronized_c_no_infect, "o-", markerfacecolor="w")
    # plt.xticks([i * c0 + c0 for i in c[::2]], rotation=30)
    # plt.yticks([i * 20 for i in range(1, 1 + int(length / 20))])
    plt.xlabel("N", fontdict=font_xy)
    plt.ylabel("no infected", fontdict=font_xy)
    plt.legend()
    with PdfPages('%sn_no_infect.pdf' % path) as pdf:
        pdf.savefig(fig)
    plt.savefig('%sn_no_infect.eps' % path, format="eps")
    plt.show()
