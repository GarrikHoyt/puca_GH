
#mcandrew
import numpy as np
import pandas as pd
from puca import puca

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import seaborn as sns

import scienceplots

import jax 

if __name__ == "__main__":

    hosp_data = pd.read_csv("./data/target-hospital-admissions.csv")
    ili_data  = pd.read_csv("./data/ili_data_all_states_2021_present__formatted.csv")

    def add_season_info(d):
        from epiweeks import Week
        from datetime import datetime
        
        time_data = {"date":[],"MMWRYR":[],"MMWRWK":[],"enddate":[],"season":[]}
        for time in d[ ["date"] ].drop_duplicates().values:
            w = Week.fromdate(datetime.strptime(time[0], "%Y-%m-%d"))
            time_data["date"].append(time[0])
            time_data["MMWRYR"].append(w.year)
            time_data["MMWRWK"].append(w.week)
            time_data["enddate"].append(w.enddate().strftime("%Y-%m-%d"))

            if w.week>=40 and w.week<=53:
                time_data["season"].append("{:d}/{:d}".format(w.year,w.year+1) )
            elif w.week>=1 and w.week<=20:
                time_data["season"].append("{:d}/{:d}".format(w.year-1,w.year) )
            else:
                time_data["season"].append("offseason")
        time_data = pd.DataFrame(time_data)

        d = d.merge(time_data, on = ["date"])
        return d

    STATE = "42" #<--these are FIPS
    
    hosp_data       = add_season_info(hosp_data)
    hosp_data       = hosp_data.loc[ (hosp_data.location==STATE) & (hosp_data.season!="offseason") ]
    hosp_data__wide = pd.pivot_table( index = ["MMWRWK"], columns = ["season"], values = ["value"], data = hosp_data )
    hosp_data__wide = hosp_data__wide.loc[ list(np.arange(40,52+1)) + list(np.arange(1,20+1))]
    
    ili_data  = ili_data.loc[ili_data.state=="pa", ["year","week","season","ili","num_patients"]]
    ili_data  = ili_data.loc[~ili_data.season.isin(["2020/2021","2021/2022"])]
    ili_data  = ili_data.rename(columns = {"year":"MMWRYR", "week":"MMWRWK"})

    ili_data__wide = pd.pivot_table( index = ["MMWRWK"], columns = ["season"], values = ["ili"], data = ili_data[["MMWRWK","season","ili"]] )
    ili_data__wide = ili_data__wide.loc[ list(np.arange(40,52+1)) + list(np.arange(1,20+1))]

    N_data__wide = pd.pivot_table( index = ["MMWRWK"], columns = ["season"], values = ["num_patients"], data = ili_data[["MMWRWK","season","num_patients"]] )
    N_data__wide = N_data__wide.loc[ list(np.arange(40,52+1)) + list(np.arange(1,20+1))]

    hosp_data__wide.columns = [y for x,y in hosp_data__wide.columns]
    ili_data__wide.columns = [y for x,y in ili_data__wide.columns]
    N_data__wide.columns    = [y for x,y in N_data__wide.columns]

    prop_hosp = 100*(hosp_data__wide/N_data__wide)
    prop_hosp = prop_hosp.loc[:, ~np.all(np.isnan(prop_hosp),0)  ]

    X                 = None                                                               #<--external covariates
    y                 = [ hosp_data__wide.to_numpy()[:,2:], ili_data__wide.to_numpy() ]    #<--stack into a list all the data needed for forecasting
    target_indicators = [ _.shape[1]-1 for _ in y ]                                        #<--last column in each dataset


    puca_model = puca(y = y , target_indicators = target_indicators, X = None)
    puca_model.fit()
    forecast   = puca_model.forecast()
    
    
    _25,_10,_250,_500,_750,_900,_975 = np.percentile( np.clip(forecast,0,None) , [2.5,10,25,50,75,90,97.5], axis=0)

    plt.style.use("science")
    fig, axs = plt.subplots(1,2)

    times = np.arange(len(_25))
    tobs  = puca_model.tobs 
    
    ax = axs[0]

    ax.fill_between(times[tobs[0]:], _25[tobs[0]:,0], _975[tobs[0]:,0], color="blue",alpha=0.10)
    ax.fill_between(times[tobs[0]:], _10[tobs[0]:,0], _900[tobs[0]:,0], color="blue",alpha=0.10)
    ax.fill_between(times[tobs[0]:], _250[tobs[0]:,0], _750[tobs[0]:,0], color="blue",alpha=0.10)
    ax.plot(times[tobs[0]:],_500[tobs[0]:,0],color="purple")

    ax.plot(hosp_data__wide["2025/2026"].values,color="black")
    ax.plot(hosp_data__wide.values,color="black",alpha=0.05)
    ax.set_ylabel("PA inc hosps")       

    ax = axs[1]
    ax.fill_between(times[tobs[1]:], _25[tobs[1]:,1], _975[tobs[1]:,1], color="blue",alpha=0.10)
    ax.fill_between(times[tobs[1]:], _10[tobs[1]:,1], _900[tobs[1]:,1], color="blue",alpha=0.10)
    ax.fill_between(times[tobs[1]:], _250[tobs[1]:,1], _750[tobs[1]:,1], color="blue",alpha=0.10)
    ax.plot(times[tobs[1]:],_500[tobs[1]:,1],color="purple")
  
    ax.plot(ili_data__wide["2025/2026"].values,color="black")
    ax.plot(ili_data__wide.values,color="black",alpha=0.05)

    ax.set_ylabel("PA ILI")
    fig.set_size_inches( (8.5-2), (11-2)/3 )
    fig.set_tight_layout(True)
    
    plt.show()


