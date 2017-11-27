import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import collections
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


def generate_simulation_hist(e_day_results, general_score, ev_states):
    """Generates historgram for election simulations

    Args:
        e_day_results (np.array): samples by state
        general_score (np.array): samples by state for general election
        ev_states (np.array): electoral votes for each state in order

    Returns:
        clinton_loses_ec_but_wins (int): number of times loses ec, wins pop
    """

    outcomes = []
    clinton_wins = 0
    clinton_loses_ec_but_wins = 0
    for i in range(10000):
        draw = np.random.randint(0, e_day_results.shape[0])
        outcome = e_day_results[draw]
        outcome = np.dot(outcome >= 0.5, ev_states)
        if outcome > 270:
            clinton_wins += 1
        else:
            if general_score[draw] > 0.5:
                clinton_loses_ec_but_wins += 1
        outcomes.append(outcome)
    clinton_loses_ec_but_wins /= 10000
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
    plt.bar(x, height, color=c)
    p = str(clinton_wins / 10000.0)
    plt.title('Probability Clinton wins = ' + p)
    plt.xlabel('Electoral Votes')
    plt.ylabel('Frequency')
    plt.show()

    return clinton_loses_ec_but_wins
