# 2021/09/27
# 画出某节点同步时间的图
# 画出横坐标为耦合强度 纵坐标为同步时间的图
# 同时画出横坐标为耦合强度 纵坐标为同步状态的图
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
import random

path = "images/node_syn_time/"
n = 32
k = 4
c=.2
synchronization_upsilon = .001 # 采用了绝对误差（因为分母过小不适合相对误差）
length = 100
n_length = 10
figsize = (5, 3.5)

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


def run_strength_time():
    inline_mat = build_network(n, k)
    xs = [0 for _ in range(3 * n)]
    paras = [0 for _ in range(n * n)]
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
    t = np.linspace(0, length, length * n_length)
    z = sol.sol(t)
    aver_x = [[0 for _ in range(len(t))] for _ in range(3)]
    synchronized_t = [[0 for _ in range(n)] for _ in range(3)]

    for j in range(3):
        for i in range(n):
            for h in range(len(t)):
                aver_x[j][h] += z.T.T[i * 3 + j][h]
        aver_x[j] = [i / n for i in aver_x[j]]
        for i in range(n):
            is_synchronized = False
            for time in range(1, len(t)):
                if abs(z[i * 3 + j][time] - aver_x[j][time]) > synchronization_upsilon:
                    is_synchronized = False
                elif not is_synchronized and time > synchronized_t[j][i]:
                    synchronized_t[j][i] = t[time]
                    is_synchronized = True
    return synchronized_t


if __name__ == '__main__':
    temp1 = run_strength_time()

    fig=plt.figure(figsize=figsize)
    plt.subplots_adjust(top=top,right=right,bottom=bottom,left=left)

    for j in range(3):
        plt.plot(range(n), temp1[j], "o-",
                 label=["S","I","R"][j], markerfacecolor="w")
    # plt.xticks([i * c0 + c0 for i in c[::2]],rotation=30)

    plt.yticks([i*20 for i in range(1,1+int(length/20))])
    plt.xlabel("n",fontdict=font_xy)
    plt.ylabel("time",fontdict=font_xy)
    plt.legend()
    with PdfPages('%snode_syn_time.pdf' % path) as pdf:
        pdf.savefig(fig)
    plt.savefig('%snode_syn_time.eps' % path, format="eps")


    plt.show()
