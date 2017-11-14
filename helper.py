import pandas as pd
import numpy as np
import tensorflow as tf
from edward.models import Normal, Binomial, MultivariateNormalFullCovariance, Uniform, Empirical
from scipy.special import logit, expit
from scipy.stats import binom


def covariance_matrix(variance, correlation, d):

    cm = variance * correlation * np.ones((d,d))
    cm += (variance - variance * correlation) * np.identity(d)

    return cm

def _sample_n(self, n, seed=None):
    # define Python function which returns samples as a Numpy array
    def np_sample(N, logits):
      p = 1 / (1 + np.exp(-1 * logits))
      return binom.rvs(N, p, random_state=seed).astype(np.float32)

    # wrap python function as tensorflow op
    # print(self.total_count)
    val = tf.py_func(np_sample, [self.total_count, self.logits], [tf.float32])[0]
    # set shape from unknown shape
    batch_event_shape = self.batch_shape.concatenate(self.event_shape)
    shape = tf.concat(
        [tf.expand_dims(n, 0), tf.convert_to_tensor(batch_event_shape)], 0)
    val = tf.reshape(val, shape)
    return val


def prepare_polls(polls, t_last):
    """Prepare the polling data by creating an index for pollsters and dates
        Dates start on day 0
        Code liberally taken from:
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

    # Taken from https://github.com/fonnesbeck/election_pycast/blob/master/Election2016.ipynb
    data_2012 = pd.read_csv('data/2012.csv', index_col=-3).sort_index()
    new_index = pd.Series(data_2012.index.values).str.lower().replace({'d.c.':'district of columbia'})
    data_2012.index = new_index

    national_score = data_2012.obama_count.sum() / (data_2012.romney_count + data_2012.obama_count).sum()

    data_2012['score'] = data_2012.obama_count / (data_2012.romney_count + data_2012.obama_count)
    data_2012['diff_score'] = data_2012.score - national_score
    data_2012['share_national'] = (data_2012.total_count * (1 + data_2012.adult_pop_growth_2011_15)
                                   / (data_2012.total_count*(1+data_2012.adult_pop_growth_2011_15)).sum())

    prior_diff_score = data_2012.diff_score
    state_weights = data_2012.share_national / data_2012.share_national.sum()
    ev_states = data_2012.ev

    return prior_diff_score, state_weights, ev_states

def predict_scores(qmu_as, qmu_bs, date_index, week_index, last_tuesday, E_day):

    t_last = np.max(date_index)
    day_2_week = {}
    for d in range(E_day + 1):
        day_2_week[d] = d // 7

    predicted_scores = []
    for day in range(E_day):
        predicted_scores.append(expit(qmu_as[day][:, np.newaxis] + qmu_bs[day_2_week[day]]))


    predicted_scores.append(expit(qmu_bs[day_2_week[E_day]]))

    # Days by samples by state
    return np.array(predicted_scores)








