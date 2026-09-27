import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("covid_19_data.csv")
data=data[data["ObservationDate"]<"07/01/2020"]
data=data[data["Country/Region"]=="Mainland China"]
data=data.drop(index=data[data['Province/State']=="Unknown"].index)

population_data=pd.read_csv("POPULATION.csv")
population_data=population_data.set_index("Province/State")

province=data["Province/State"].drop_duplicates().values


max_infect_list=[]
max_infect_distribution=[]

for i in province:
    data_temp=data[data["Province/State"]==i]
    #population_province=population_data[population_data["Province/State"]==i]["Population"].values()
    population_province=population_data.at[i,"Population"]
    max_infect_list.append(max(data_temp["Confirmed"].values-
                               data_temp["Deaths"].values-
                               data_temp["Recovered"].values)
                           /population_province)
print(max_infect_list)
plt.plot(sorted(max_infect_list))
plt.show()