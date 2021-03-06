import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import collections
import pandas as pd
from helper import predict_scores, get_brier_score
from scipy.special import expit
plt.style.use('ggplot')

START_DATE = dt.date(2016, 5, 1)
ELECTION_DATE = dt.date(2016, 11, 8)


def generate_time_plot(state_scores, state_s_polls, burn_in, state_name, prior,
                       save=False):
    """Generates a time plot

    Args:
        states_scores (np.array): states by samples
        state_s_polls (pd.Dataframe): polls for state s
        burn_in (int): number of samples to throw out
        state_name (str): name of the state
        prior (float): prior on election day
        save (bool, optional): whether to save the plot
    """
    n_days = state_scores.shape[0]
    x_coord = np.arange(n_days)
    burn_state_scores = state_scores[:, burn_in:]

    # plot dots (polls)
    poll_date = state_s_polls.date_index.as_matrix()
    poll_value = state_s_polls.p_clinton.as_matrix()
    plt.scatter(poll_date, poll_value, s=10, c='black')

    # plot thick blue line (median) and 90% confidence interval
    medians = np.median(burn_state_scores, axis=1)
    uppers = np.percentile(burn_state_scores, 95, axis=1)
    lowers = np.percentile(burn_state_scores, 5, axis=1)

    plt.title('Clinton Vote Share over Time for {0}, P(Win) = {1:.4f}'.format(
        state_name.title(), np.mean(burn_state_scores[-1] > 0.5)), fontsize=11)
    plt.ylabel('Share of Two-Party Vote')
    plt.xlabel('Day')
    plt.plot(x_coord, medians, color='blue')
    plt.plot(x_coord, lowers, color='#33CEFF')
    plt.plot(x_coord, uppers, color='#33CEFF')
    plt.fill_between(x_coord, uppers, lowers,
                     interpolate=True, color='#33CEFF', alpha=.4)
    plt.axhline(y=0.5, color='black', linestyle='-')
    plt.axhline(y=prior, color='black', linestyle='--')

    if state_name != 'general':
        results_2016 = pd.read_csv(
            '../data/2016_results.csv', index_col=0)
        results_2016 = results_2016['dem_share_2p']
        plt.axhline(y=results_2016[state_name], color='purple', linestyle=':')
    else:
        plt.axhline(y=0.511, color='purple', linestyle=':')

    if save is True:
        plt.savefig('../plots/time_plots/' +
                    state_name.replace(" ", "_") + '.png', dpi=300)
        plt.clf()


def generate_undecided_plot(undecided_table, state_index, state_name, mean_w,
                            mean_b, E_day, save=False):
    """Generates undecided plot

    Args:
        undecided_table (np.array): # of und for state, day
        state_index (int): index of state
        state_name (str): name of state
        mean_w (np.array): slope for und line
        mean_b (np.array): intercept
        E_day (int): index for election day
    """
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


def generate_house_effects_hist(qmu_c):

    median = np.median(qmu_c, axis=0)
    median = expit(median) - 0.5
    blue = median[np.where(median >= 0)[0]]
    red = median[np.where(median < 0)[0]]
    bins = np.arange(-0.032, 0.032, 0.002)
    plt.figure(figsize=(11, 7))
    n, bins, pathces = plt.hist(blue, bins=bins, color='blue', alpha=0.7)
    n, bins, pathces = plt.hist(red, bins=bins, color='red', alpha=0.7)
    plt.xlabel('Approximate House Effects')
    plt.ylabel('Number of Pollsters')
    plt.title('House Effects')
    plt.show()


def generate_simulation_hist(e_day_results, general_score, ev_states,
                             graph=True, sim=10000):
    """Generates historgram for election simulations

    Args:
        e_day_results (np.array): samples by state
        general_score (np.array): samples by state for e_day general election
        ev_states (np.array): electoral votes for each state in order

    Returns:
        clinton_loses_ec_but_wins (int): number of times loses ec, wins pop
    """

    outcomes = []
    clinton_wins = 0
    clinton_loses_ec_but_wins = 0
    for i in range(sim):
        draw = np.random.randint(0, e_day_results.shape[0])
        outcome = e_day_results[draw]
        outcome = np.dot(outcome >= 0.5, ev_states)
        if outcome > 270:
            clinton_wins += 1
        else:
            if general_score[draw] > 0.5:
                clinton_loses_ec_but_wins += 1
        outcomes.append(outcome)
    clinton_loses_ec_but_wins /= sim
    p = str(clinton_wins / sim)

    if graph is True:
        x = np.unique(outcomes)
        freq = collections.Counter(outcomes)
        height = [freq[s] for s in x]
        c = []
        for out in x:
            if out > 270:
                c.append('blue')
            else:
                c.append('red')
        plt.figure(figsize=(15, 7))
        plt.bar(x, height, color=c, alpha=0.7)
        plt.title('Probability Clinton wins = ' + p)
        plt.xlabel('Electoral Votes')
        plt.ylabel('Frequency')
        plt.show()

    return clinton_loses_ec_but_wins, p


def generate_state_probs(states, e_day_scores):

    results_2016 = pd.read_csv(
        '../data/2016_results.csv', index_col=0)
    results_2016 = results_2016['win']
    results_2016 = results_2016.loc[states].as_matrix().flatten()

    probabilities = np.mean(e_day_scores > 0.5, axis=0)
    argsort = np.argsort(probabilities)

    c = []
    for p in probabilities[argsort]:
        if p > 0.5:
            c.append('blue')
        else:
            c.append('red')

    y = np.array(states)
    plt.figure(figsize=(7, 10))
    plt.scatter(probabilities[argsort], np.arange(51), color=c)
    plt.yticks(np.arange(51), y[argsort])
    plt.title("State Probabilities")
    plt.xlabel("Probability Clinton Wins State")
    plt.show()


def variance_test(qmu_as, qmu_bs, E_day, mean_w, mean_b, state_weights_np,
                  ev_states, states, BURN_IN):

    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()
    var = [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1]
    for v in var:
        predicted_scores = predict_scores(
            qmu_as, qmu_bs, E_day, mean_w, mean_b, var=v)
        general_score = np.sum(state_weights_np * predicted_scores, axis=2)
        clean_scores = predicted_scores[:, BURN_IN:, :]
        e_day_results = clean_scores[-1, :, :]
        e_day_general = general_score[-1, BURN_IN:]

        clinton_loses_ec_but_wins, clinton_wins = generate_simulation_hist(
            e_day_results, e_day_general, ev_states, graph=False)
        ax1.scatter(v, clinton_wins, s=50, c='orange')
        brier_score = get_brier_score(e_day_results, states)
        ax2.scatter(v, brier_score, c="purple")

    plt.title("Changing the variance of undecided voters")
    plt.xlabel("Variance of Logit Normal")
    ax1.set_xlabel("Variance")
    ax1.set_ylabel("Probability Clinton Wins", color='orange')
    ax1.tick_params('y', colors='orange')
    ax2.set_ylabel("Evenly Weighted Brier Score", color='purple')
    ax2.tick_params('y', colors='purple')
    plt.show()
