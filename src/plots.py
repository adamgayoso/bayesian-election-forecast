# %load plots.py
import pandas as pd
import numpy as np
import edward as ed
import tensorflow as tf
import datetime as dt
import random
from scipy import stats
#from ggplot import *
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
plt.style.use('ggplot')

START_DATE = dt.date(2016, 5, 1)
ELECTION_DATE = dt.date(2016, 11, 8)


def generate_time_plot(state_scores, state_s_polls, burn_in, state_name, prior, save=False):

    medians = []

    means = []
    lowers = []
    uppers = []
    sigmas = []
    x_coord_date = []
    x_coord = []
    for day in range(0, 191):
        x_coord.append(day)
        # x_coord_date.append()
    # median of posterior - thick blue
    # light-blue region = 90% credible interval
    # thin-blue lines = 100 draws from posterior

    samples_chosen = 100
    sample_size = 10000
    my_randoms = random.sample(range(burn_in, sample_size), samples_chosen)
    burn_state_scores = state_scores[:, burn_in:]

    # plot thin blue lines (100 posterior draws)
    # for r in my_randoms:
    #     y_coord=[]
    #     for day in range(0,191):
    #         y_coord.append(state_scores[day][r])
    #     plt.plot(x_coord,y_coord,linewidth=0.05)
    # plot 90% confidence interval
    x_conf = []
    y_conf = []

    # plot dots (polls)
    index_x = ([])
    index_y = ([])
    for po_pc in (state_s_polls.p_clinton):
        index_y.append(po_pc)
    for po_date in (state_s_polls.poll_date):
        time_from_start = abs(po_date.date() - START_DATE)
        index_x.append(time_from_start.days)
    sizes_s = [10 for n in range(len(index_x))]
    plt.scatter(index_x, index_y, s=sizes_s, c='black')

    # plot thick blue line (median) and 90% confidence interval
    for day in range(0, 191):
        day_scores = burn_state_scores[day]
        day_median = np.median(day_scores)
        day_mean, day_sigma = np.mean(day_scores), np.std(day_scores)
        conf_int = stats.norm.interval(0.9, loc=day_mean, scale=day_sigma)
        lowers.append(conf_int[0])
        uppers.append(conf_int[1])
        medians.append(day_median)
        sigmas.append(day_sigma)
        means.append(day_mean)
    plt.ylabel('Pr(Clinton wins)' + state_name)
    plt.plot(x_coord, medians, color='blue')
    plt.plot(x_coord, lowers, color='#33CEFF')
    plt.plot(x_coord, uppers, color='#33CEFF')
    plt.fill_between(x_coord, uppers, lowers,
                     interpolate=True, color='#33CEFF', alpha=.4)
    plt.axhline(y=0.5, color='black', linestyle='-')
    plt.axhline(y=prior, color='black', linestyle='--')


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
