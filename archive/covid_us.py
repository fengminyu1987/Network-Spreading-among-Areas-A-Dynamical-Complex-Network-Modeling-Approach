import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    max_infect_list.append(max(data_temp["Confirmed"].values -
                               data_temp["Deaths"].values -
                               data_temp["Recovered"].values) / population_province)
print(max_infect_list)
# plt.hist(sorted(max_infect_list), 20, density=1, facecolor='g', alpha=0.75)
plt.bar(range(len(max_infect_list)), sorted(max_infect_list), alpha=0.5, color="grey")
plt.show()
