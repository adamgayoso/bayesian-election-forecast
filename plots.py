import pandas as pd
import numpy as np
import edward as ed
import tensorflow as tf
import datetime as dt
import random
#from ggplot import *
import matplotlib.pyplot as plt

START_DATE = dt.date(2016, 5, 1)
ELECTION_DATE = dt.date(2016, 11, 8)

def generate_plot(state_scores, this_state_polls, burn_in, state_name):
    
    medians=[]
    x_coord=[]
    for day in range(0,191):
        x_coord.append(day)
    #median of posterior - thick blue
    #light-blue region = 90% credible interval
    #thin-blue lines = 100 draws from posterior
    
    samples_chosen = 100
    sample_size = 10000
    my_randoms = random.sample(range(burn_in, sample_size), samples_chosen)
    burn_state_scores = state_scores[:,burn_in:]
    
    #plot thin blue lines (100 posterior draws)
    for r in my_randoms:
        y_coord=[]
        for day in range(0,191):
            y_coord.append(state_scores[day][r])
        plt.plot(x_coord,y_coord,linewidth=0.05)
    #plot 90% confidence interval
            
    #plot dots (polls)
    
    #plot thick blue line (median)
    for day in range(0,191):
        day_scores = burn_state_scores[day]
        day_median = np.median(day_scores)
        medians.append(day_median)
    plt.ylabel('Pr(Clinton wins)'+state_name)
    plt.plot(x_coord,medians)
    
    plt.show()