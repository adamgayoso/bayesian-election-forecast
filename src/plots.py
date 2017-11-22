# %load plots.py
import pandas as pd
import numpy as np
import datetime as dt
import random
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import collections
plt.style.use('ggplot')

START_DATE = dt.date(2016, 5, 1)
ELECTION_DATE = dt.date(2016, 11, 8)


def generate_time_plot(state_scores, state_s_polls, burn_in, state_name, prior, save=False):

    n_days = state_scores.shape[0]
    x_coord = np.arange(n_days)
    # x_coord_date.append()
    # median of posterior - thick blue
    # light-blue region = 90% credible interval
    # thin-blue lines = 100 draws from posterior

    # samples_chosen = 100
    # sample_size = 10000
    # my_randoms = random.sample(range(burn_in, sample_size), samples_chosen)
    burn_state_scores = state_scores[:, burn_in:]

    # plot thin blue lines (100 posterior draws)
    # for r in my_randoms:
    #     y_coord=[]
    #     for day in range(0,n_days):
    #         y_coord.append(state_scores[day][r])
    #     plt.plot(x_coord,y_coord,linewidth=0.05)
    # plot 90% confidence interval

    # plot dots (polls)
    poll_date = state_s_polls.date_index.as_matrix()
    poll_value = state_s_polls.p_clinton.as_matrix()
    plt.scatter(poll_date, poll_value, s=10, c='black')

    # plot thick blue line (median) and 90% confidence interval
    medians = np.median(burn_state_scores, axis=1)
    uppers = np.percentile(burn_state_scores, 95, axis=1)
    lowers = np.percentile(burn_state_scores, 5, axis=1)

    plt.title('Clinton Vote Share over Time for ' + state_name.title())
    plt.ylabel('Share of Two-Party Vote')
    plt.xlabel('Day')
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
    state_name = state_name.title()
    date_range = np.arange(E_day)
    state = undecided_table[np.where(undecided_table[:, 2] == s)[0]]
    dates = state[:, 1]
    und = state[:, 0]
    plt.scatter(dates, und)
    plt.title('Undecided Voters in ' + state_name.title())
    plt.xlabel('Day index')
    plt.ylabel('Percantage Undecided')
    plt.plot(date_range, mean_w[s] * date_range + mean_b[s], c='orange')

    if save is True:
        plt.savefig('../plots/undecided_plots/' +
                    state_name.replace(" ", "_") + '.png', dpi=300)
        plt.clf()


def generate_simulation_hist(outcomes, clinton_wins):
    x = np.unique(outcomes)
    freq = collections.Counter(outcomes)
    height = [freq[s] for s in x]
    c = []
    for out in x:
        if out > 270:
            c.append('blue')
        else:
            c.append('red')
    plt.figure(figsize=(15,7))
    plt.bar(x, height, color=c)
    p = str(clinton_wins / 10000.0)
    plt.title('Probability Clinton wins = ' + p)
    plt.xlabel('Electoral Votes')
    plt.ylabel('Frequency')
    plt.show()
