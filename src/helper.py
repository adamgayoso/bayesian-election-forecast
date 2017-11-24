import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.special import expit
from scipy.stats import binom


def covariance_matrix(variance, correlation, d):

    cm = variance * correlation * np.ones((d, d))
    cm += (variance - variance * correlation) * np.identity(d)

    return cm


def _sample_n(self, n, seed=None):
    # define Python function which returns samples as a Numpy array
    def np_sample(N, logits):
        p = 1 / (1 + np.exp(-1 * logits))
        return binom.rvs(N, p, random_state=seed).astype(np.float32)

    # wrap python function as tensorflow op
    # print(self.total_count)
    val = tf.py_func(np_sample, [self.total_count,
                                 self.logits], [tf.float32])[0]
    # set shape from unknown shape
    batch_event_shape = self.batch_shape.concatenate(self.event_shape)
    shape = tf.concat(
        [tf.expand_dims(n, 0), tf.convert_to_tensor(batch_event_shape)], 0)
    val = tf.reshape(val, shape)
    return val


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


def get_brier_score(e_day_scores, state_polls):

    results_2016 = pd.read_csv('../data/2016_results.csv', index_col=0, header=None)
    results_2016 = results_2016.loc[state_polls.state.unique()].as_matrix().flatten()

    probabilities = np.mean(e_day_scores > 0.5, axis=0)

    brier_score = np.sum(np.square(results_2016 - probabilities))
    brier_score /= len(results_2016)

    return brier_score


