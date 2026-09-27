# 画出sir的折线图
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
import random

path="images/synchronize_linechart/sir/"
H_vec = [1, 0.01, 1]
beta_gamma=[.5,.2]
s_i_r_init=[.9999,.0001,0]
bottom=.2
top=.98
right=.98
left=.28
text_max_min=[-.005,-.03,17]

n = 16
k = 4
c = .3

font_title = {'family': 'Times New Roman',
        'weight': 'bold',
		'style':'italic',
        'size': 18,
        }
font_xy = {'family': 'Times New Roman',
        'weight': 'normal',
		'style':'italic',
        'size': 30,
           }
font_axis = {'family': 'Times New Roman',
        'weight': 'normal',
		'style':'italic',
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
    g = nx.barabasi_albert_graph(n, k, seed=123)
    fig_nw = plt.figure(figsize=figsize)
    nx.draw_networkx(g, pos=nx.spring_layout(g),**{"node_color": "pink",
    "edgecolors": "black"})
    with PdfPages('%snetwork.pdf'%path) as pdf:
        pdf.savefig(fig_nw)
    plt.savefig('%snetwork.eps'%path, format="eps")

    inline_mat = [[0 for i in range(n)] for j in range(n)]

    for i in g.edges:
        strength = 1
        inline_mat[i[0]][i[1]] = strength
        inline_mat[i[1]][i[0]] = strength
    for i in range(len(inline_mat)):
        inline_mat[i][i] = -sum(inline_mat[i])
    [print(i) for i in inline_mat]
    [print("%d=%d" % (i, g.degree(i)), end=" ") for i in g.nodes]
    return inline_mat


def f1(x):
    if x % 3 == 0:
        return "S$_%d$" % (x / 3)
    elif x % 3 == 1:
        return "I$_%d$" % (x / 3)
    else:
        return "R$_%d$" % (x / 3)

def rate_infection(z):
    result=[]
    for i in range(n):
        result.append(max(z[i * 3 + 1]))
    return sorted(result)


if __name__ == '__main__':

    length=100
    figsize=(5,4.2)
    inline_mat = build_network(n, k)
    xs = [0 for i in range(3 * n)]
    paras = [0 for i in range(n * n)]
    for i in range(n):
        xs[i * 3] = s_i_r_init[0]
        xs[i * 3 + 1] = 0
        xs[i * 3 + 2] = 1-s_i_r_init[0]
        for j in range(n):
            paras[i * n + j] = c * inline_mat[i][j]
    xs[1] = s_i_r_init[1]
    xs[2]=  s_i_r_init[2]

    sol = solve_ivp(sir, [0, length], xs, args=paras, method='Radau',
                    dense_output=True)
    t = np.linspace(0, length, 30)
    z = sol.sol(t)
    sol_isolate = solve_ivp(sir, [0, length], [(xs[0]+1*n-1)/n, xs[1]/n, xs[2]/n],
                            args=[paras[0], paras[1], 0],
                            method='Radau',
                            dense_output=True)
    z_isolate = sol_isolate.sol(t)


    fig0=plt.figure(figsize=figsize)
    plt.subplots_adjust(bottom=bottom, top=top, right=right)
    plt.plot(t, z.T, "-o", markersize=5, markerfacecolor="w")
    plt.xlabel('t',fontdict=font_xy)
    with PdfPages('%ssir.pdf'%path) as pdf:
        pdf.savefig(fig0)
    plt.savefig('%ssir.eps'%path, format="eps")
    # plt.title('SIR System')

    aver_x = [[0 for i in range(len(t))] for j in range(3)]
    for j in range(3):
        fig=plt.figure(figsize=figsize)
        plt.subplots_adjust(left=left, bottom=bottom,top=top,right=right)
        for i in range(n):
            plt.plot(t, z.T.T[i * 3 + j]/(z.T.T[i * 3 ]+z.T.T[i * 3 + 1]+z.T.T[i * 3 + 2]), "-o", markersize=5, markerfacecolor="w")
            for k in range(len(t)):
                aver_x[j][k] += z.T.T[i * 3 + j][k]
        plt.xlabel('t', fontdict=font_xy)
        plt.xticks(fontproperties=font_axis)
        plt.yticks(fontproperties=font_axis)
        plt.ylabel("%s(t)" % ["S", "I", "R"][j], fontdict=font_xy)
        aver_x[j]=[i / n for i in aver_x[j]]
        plt.plot(t, aver_x[j], "-o", linestyle="dotted", color="black",
                 markersize=5, markerfacecolor="w")
        print("%s %.2f" % (["S", "I", "R"][j],aver_x[j][-1]))
        if j==2 or j==0:
            plt.text(text_max_min[2],aver_x[j][-1]+(text_max_min[0] if aver_x[j][-1]<.5 else text_max_min[1]),
                     "%.2f"%aver_x[j][-1],ha="center",
                     backgroundcolor="w",color="black",fontsize=14)
            plt.plot(t, [aver_x[j][-1] for i in t], "-.", color="gray",
                    markersize=5, markerfacecolor="w")
        plt.ylim(-.05,1.05)
        # plt.plot(t, z_isolate.T.T[j], "-o", linestyle="dotted", color="black",
        #          markersize=5, markerfacecolor="w")

        # plt.legend(["%s$_{%02d}$" % (["S", "I", "R"][j], i) for i in range(n)], ncol=5)

        # plt.title(["S", "I", "R"][j],fontdict=font_title)
        with PdfPages('%ssir_%s.pdf'%(path,["S", "I", "R"][j])) as pdf:
            pdf.savefig(fig)
        plt.savefig('%ssir_%s.eps'%(path,["S", "I", "R"][j]), format="eps")

    # aver_x = [[i / n for i in aver_x[j]] for j in range(3)]


    for j in range(3):
        fig=plt.figure(figsize=figsize)
        plt.subplots_adjust(left=left, bottom=bottom,top=top,right=right)
        for i in range(n):
            plt.plot(t, z.T.T[i * 3 + j] - aver_x[j], "-o", markersize=5, markerfacecolor="w")
        plt.xlabel('t', fontdict=font_xy)
        plt.ylabel("$e_%s$(t)" % ["S", "I", "R"][j], fontdict=font_xy)
        plt.ylim(-0.2,0.65)
        plt.xticks(fontproperties=font_axis)
        plt.yticks(fontproperties=font_axis)
        # plt.plot(t, z_isolate.T.T[j] - aver_x[j], "-o", linestyle="dotted", color="black",
        #          markersize=5, markerfacecolor="w")

        #plt.legend(["%s$_{%02d}$" % (["S", "I", "R"][j], i) for i in range(n)], ncol=5)

        # plt.title(["S", "I", "R"][j],fontdict=font_title)
        with PdfPages('%ssir_err_%s.pdf'%(path,["S", "I", "R"][j])) as pdf:
            pdf.savefig(fig)
        plt.savefig('%ssir_err_%s.eps'%(path,["S", "I", "R"][j]), format="eps")

    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(left=left, bottom=bottom, top=top, right=right)
    plt.plot(range(n),rate_infection(z))

    plt.show()
