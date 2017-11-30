import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.special import expit


def covariance_matrix(variance, correlation, d):
    """Creates a covariance matrix

    Args:
        variance (float): value of diagonal
        correlation (float): correlation between variables
        d (int): dimension of square matrix

    Returns:
        np.array: d by d covariance matrix
    """
    cm = variance * correlation * np.ones((d, d))
    cm += (variance - variance * correlation) * np.identity(d)

    return cm


def prepare_polls(polls, t_last):
    """Prepare the polling data by creating an index for pollsters and dates
        Dates start on day 0
        Code modified from:
        https://github.com/fonnesbeck/election_pycast/blob/master/Election2016.ipynb

    Args:
        polls (DataFrame): raw all_polls DF from HuffPo api
        t_last (date): day up until you would like polls

    Returns:
        tuple: state polls, national polls
    """
    polls = polls.copy()
    rows = polls.poll_date <= t_last
    polls = polls.loc[rows]

    polls.loc[:, 'week_index'] = polls.week - polls.week.min()
    days = pd.date_range(polls.poll_date.min(),
                         polls.poll_date.max())
    days = pd.Series(range(len(days)), index=days)

    # Integer to represent day for each poll in polls
    poll_2_dayID = days.loc[polls.poll_date]

    # Assign an ID to each pollster for each poll in polls
    pollsters = polls.pollster.unique()
    enumerated_pollsters = pd.Series(range(len(pollsters)), index=pollsters)
    poll_2_pollsterID = enumerated_pollsters.loc[polls.pollster]

    polls.loc[:, 'date_index'] = poll_2_dayID.values
    polls.loc[:, 'pollster_index'] = poll_2_pollsterID.values

    national_poll_inds = polls.state == 'general'
    national_polls = polls.loc[national_poll_inds]
    state_polls = polls.loc[~national_poll_inds]

    # Assign an ID to each state
    states = state_polls.state.unique()
    enumerated_states = pd.Series(range(len(states)), index=states)
    poll_2_stateID = enumerated_states.loc[state_polls.state]
    state_polls.loc[:, 'state_index'] = poll_2_stateID.values

    return state_polls, national_polls


def process_2012_polls():
    """Process the 2012 election results
        Code modified from:
        https://github.com/fonnesbeck/election_pycast/blob/master/Election2016.ipynb
        Data from:
        https://github.com/pkremp/polls

    Returns:
        series: difference between state vote and national vote for 2012
        series: population weight of each state
        series: electoral votes for each state
    """
    data_2012 = pd.read_csv('../data/2012.csv', index_col=-3).sort_index()
    new_index = pd.Series(data_2012.index.values).str.lower(
    ).replace({'d.c.': 'district of columbia'})
    data_2012.index = new_index

    national_score = data_2012.obama_count.sum()
    national_score /= (data_2012.romney_count + data_2012.obama_count).sum()

    data_2012['score'] = data_2012.obama_count / \
        (data_2012.romney_count + data_2012.obama_count)
    data_2012['diff_score'] = data_2012.score - national_score
    apg1115 = data_2012.adult_pop_growth_2011_15
    tc = data_2012.total_count
    share_national = (tc * (1 + apg1115) / (tc * (1 + apg1115)).sum())
    data_2012['share_national'] = share_national

    prior_diff_score = data_2012.diff_score
    state_weights = data_2012.share_national / data_2012.share_national.sum()
    ev_states = data_2012.ev

    return prior_diff_score, state_weights, ev_states


def predict_scores(qmu_as, qmu_bs, E_day, w, b, var=0.25):
    """Predicts daily vote intentions using results from inference
    Args:
        qmu_as (np.array): mu_a posterior samples
        qmu_bs (np.array): mu_b posterior samples
        E_day (int): index for election day

    Returns:
        np.array: days by samples by states score
    """
    # t_last = np.max(date_index)
    n_samples, n_states = qmu_bs[0].shape
    day_2_week = {}
    for d in range(E_day + 1):
        day_2_week[d] = d // 7

    sigma_poll_error = covariance_matrix(var, 0.75, n_states)
    predicted_scores = []
    e = np.random.multivariate_normal(
        np.zeros(n_states), cov=sigma_poll_error, size=n_samples)

    # Logistic-Normal Transformation
    exp_e = expit(e)

    for day in range(E_day + 1):
        und = (w * day + b) / 100
        und_c = exp_e * und
        if day != E_day:
            p = expit(qmu_as[day][:, np.newaxis] + qmu_bs[day_2_week[day]])
            predicted_scores.append(p * (1 - und) + und_c)
        else:
            p = expit(qmu_bs[day_2_week[E_day]])
            predicted_scores.append(p * (1 - und) + und_c)

    # Days by samples by state
    return np.array(predicted_scores)


def get_brier_score(e_day_scores, states, weighted=False, ev_states=None):
    """Calculates brier score for election day results
    Args:
        e_day_scores (np.array): samples by states
        states (pd.Series): unique states in state_polls

    Returns:
        float: brier score
    """

    results_2016 = pd.read_csv(
        '../data/2016_results.csv', index_col=0)
    results_2016 = results_2016['win']
    results_2016 = results_2016.loc[states].as_matrix().flatten()

    probabilities = np.mean(e_day_scores > 0.5, axis=0)

    if weighted is True:
        brier_score = np.sum(
            ev_states * np.square(results_2016 - probabilities))
        brier_score /= np.sum(ev_states)

    else:
        brier_score = np.sum(np.square(results_2016 - probabilities))
        brier_score /= len(results_2016)

    return brier_score


def get_correct(e_day_scores, states, print=False):
    results_2016 = pd.read_csv(
        '../data/2016_results.csv', index_col=0)
    results_2016 = results_2016['win']
    results_2016 = results_2016.loc[states].as_matrix().flatten()

    probabilities = np.mean(e_day_scores > 0.5, axis=0)
    probabilities = np.around(probabilities)
    correct = probabilities == results_2016
    if print is True:
        for i in range(len(correct)):
            if correct[i] == 0:
                print(states[i])
    correct = np.sum(correct)
    return correct


def brier_test(e_day_results, states, ev_states):

    brier_score = get_brier_score(e_day_results, states)
    weighted_brier_score = get_brier_score(
        e_day_results, states, weighted=True, ev_states=ev_states)
    num_correct = get_correct(e_day_results, states)
    print("Brier Score: {0:.4f}".format(brier_score))
    print("EC Weighted Brier Score: {0:.4f}".format(weighted_brier_score))
    print("Number Correct (Incl. DC): {0}".format(num_correct))


def assemble_polls(mu_bs, mu_as, mu_a_buffer, mu_c, national_polls,
                   state_polls, state_weights, E_day, E_week):

    mu_b_tf = tf.stack(mu_bs)
    mu_a_tf = tf.stack(mu_as)
    mu_a_tf = tf.concat([mu_a_buffer, mu_a_tf], axis=0)
    # Due to list in reverse
    mu_a_state = tf.gather(
        mu_a_tf, (E_day - state_polls.date_index).as_matrix())
    state_ind = state_polls[['week_index', 'state_index']].as_matrix()
    # Due to list in reverse
    state_ind[:, 0] = E_week - state_ind[:, 0]

    mu_b_state = tf.gather_nd(mu_b_tf, state_ind)
    mu_c_state = tf.gather(mu_c, state_polls.pollster_index)

    state_logits = mu_b_state + mu_a_state

    nat_ind = national_polls[['week_index', 'date_index']].as_matrix()
    # Due to list in reverse
    nat_ind[:, 0] = E_week - nat_ind[:, 0]
    nat_ind[:, 1] = E_day - nat_ind[:, 1]
    mu_b_nat = tf.gather(mu_b_tf, nat_ind[:, 0])
    mu_a_nat = tf.expand_dims(tf.gather(mu_a_tf, nat_ind[:, 1]), 1)
    # expit
    nat_expits = 1 / (1 + tf.exp(-1 * (mu_a_nat + mu_b_nat)))
    # logit
    nat_weigh_avg = tf.multiply(state_weights, nat_expits)
    nat_weigh_avg = -tf.log((1 / (tf.reduce_sum(nat_weigh_avg, axis=1))) - 1)
    mu_c_nat = tf.gather(mu_c, national_polls.pollster_index)

    final_logits = tf.concat([state_logits, nat_weigh_avg], axis=0)
    final_logits += tf.concat([mu_c_state, mu_c_nat], axis=0)

    return final_logits


def extract_results(mu_as, mu_bs, mu_c, inference):

    qmu_bs = []
    for b in mu_bs:
        qmu_bs.append(inference.latent_vars[b].params.eval())
    qmu_bs = list(reversed(qmu_bs))

    qmu_as = []
    for a in mu_as:
        qmu_as.append(inference.latent_vars[a].params.eval())
    qmu_as = list(reversed(qmu_as))

    qmu_c = inference.latent_vars[mu_c].params.eval()

    return qmu_as, qmu_bs, qmu_c
