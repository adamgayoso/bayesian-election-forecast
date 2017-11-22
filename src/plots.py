import pandas as pd
import numpy as np
import edward as ed
import tensorflow as tf
import datetime as dt
import random
#from ggplot import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')

START_DATE = dt.date(2016, 5, 1)
ELECTION_DATE = dt.date(2016, 11, 8)


def generate_time_plot(state_scores, this_state_polls, burn_in, state_name, save=False):

    medians = []
    x_coord = []
    for day in range(0, 191):
        x_coord.append(day)
    # median of posterior - thick blue
    # light-blue region = 90% credible interval
    # thin-blue lines = 100 draws from posterior

    samples_chosen = 100
    sample_size = 10000
    my_randoms = random.sample(range(burn_in, sample_size), samples_chosen)
    burn_state_scores = state_scores[:, burn_in:]

    # plot thin blue lines (100 posterior draws)
    for r in my_randoms:
        y_coord = []
        for day in range(0, 191):
            y_coord.append(state_scores[day][r])
        plt.plot(x_coord, y_coord, linewidth=0.05)
    # plot 90% confidence interval

    # plot dots (polls)

    # plot thick blue line (median)
    for day in range(0, 191):
        day_scores = burn_state_scores[day]
        day_median = np.median(day_scores)
        medians.append(day_median)
    plt.ylabel('Pr(Clinton wins)' + state_name)
    plt.plot(x_coord, medians)

    if save is True:
        plt.savefig('../plots/time_plots/' +
                    state_name.replace(" ", "_") + '.png', dpi=300)
        plt.clf()


def generate_undecided_plot(undecided_table, state_index, state_name, mean_w, mean_b, E_day, save=False):

    s = state_index
    date_range = np.arange(E_day)
    state = undecided_table[np.where(undecided_table[:, 2] == s)[0]]
    dates = state[:, 1]
    und = state[:, 0]
    plt.scatter(dates, und)
    plt.title('Undecided Voters in ' + state_name)
    plt.xlabel('Day index')
    plt.ylabel('Percantage Undecided')
    plt.plot(date_range, mean_w[s] * date_range + mean_b[s], c='orange')

    if save is True:
        plt.savefig('../plots/undecided_plots/' +
                    state_name.replace(" ", "_") + '.png', dpi=300)
        plt.clf()
