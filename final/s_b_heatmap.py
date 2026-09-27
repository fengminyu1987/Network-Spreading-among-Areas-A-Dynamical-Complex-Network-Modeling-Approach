# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 21:17:12 2022

@author: ASUS
"""
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx
import random
import json
import time
import pandas as pd
import os
import seaborn as sns

font_xy = {
    "family": "Times New Roman",
    "weight": "normal",
    "style": "italic",
    "size": 22
}
font_axis = {
    "family": "Times New Roman",
    "weight": "normal",
    "style": "italic",
    "size": 15
}

path = "D:/PythonProjects/my/final/images/strength_beta/WS/"
data = json.load(open("%sdata.json" % path, "r"))
data_dict = {}
cs = data["cs"]
betas = data["betas"]
# %% 按照平均值
fig = plt.figure(figsize=(5.4, 3.8))
plt.subplots_adjust(left=.18, top=.96, bottom=.2)
data_dict["betas"] = betas
for i in cs:
    data_dict[i] = []
    for j in betas:
        data_dict[i].append(data["infection_ratio"][str(i)][str(j)][1])
print(data_dict)
df = pd.DataFrame(data_dict)
df = df.set_index(["betas"])
df = df.sort_index(ascending=False)

sns.heatmap(data=df, cmap=plt.get_cmap('winter'),
            xticklabels=2, yticklabels=2, linewidth=0.1)
plt.xticks(rotation=45, fontproperties=font_axis)
plt.yticks(rotation=45, fontproperties=font_axis)
plt.ylabel(r"$\beta$", fontdict=font_xy)
plt.xlabel(r"$\sigma$", fontdict=font_xy)

with PdfPages('%smean.pdf' % path) as pdf:
    pdf.savefig(fig)
plt.savefig('%smean.eps' % path, format="eps")

# %% 按照最大值
fig = plt.figure(figsize=(5.4, 3.8))
plt.subplots_adjust(left=.18, top=.96, bottom=.2)
data_dict["betas"] = betas
for i in cs:
    data_dict[i] = []
    for j in betas:
        data_dict[i].append(data["infection_ratio"][str(i)][str(j)][-1])

df = pd.DataFrame(data_dict)
df = df.set_index(["betas"])
df = df.sort_index(ascending=False)

sns.heatmap(data=df, cmap=plt.get_cmap('summer'),
            xticklabels=2, yticklabels=2, linewidth=0.1)
plt.xticks(rotation=45, fontproperties=font_axis)
plt.yticks(rotation=45, fontproperties=font_axis)
plt.ylabel(r"$\beta$", fontdict=font_xy)
plt.xlabel(r"$\sigma$", fontdict=font_xy)

with PdfPages('%smax.pdf' % path) as pdf:
    pdf.savefig(fig)
plt.savefig('%smax.eps' % path, format="eps")
