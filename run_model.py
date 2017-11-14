import pandas as pd
import numpy as np
import edward as ed
import tensorflow as tf
from edward.models import Normal, NormalWithSoftplusScale, Binomial, MultivariateNormalFullCovariance, Uniform, Empirical, InverseGamma, Exponential
from scipy.special import logit, expit
import datetime as dt
# import math
from scipy.stats import binom
from helper import _sample_n, prepare_polls, process_2012_polls, predict_scores, covariance_matrix
import matplotlib.pyplot as plt
import collections

ELECTION_DATE = dt.date(2016, 11, 8)
BURN_IN = 3000
pd.options.mode.chained_assignment = None

def main():

    # Load data
    polls = pd.read_csv('data/all_polls_2016.csv', parse_dates=['begin', 'end', 'poll_date'])
    up_to_t = dt.date(2016, 11, 8)
    state_polls, national_polls = prepare_polls(polls, up_to_t)

    # Get prior information from 2012 election
    prior_diff_score, state_weights, ev_states = process_2012_polls()
    prior_diff_score = prior_diff_score[state_polls.state.unique()]
    state_weights = state_weights[state_polls.state.unique()].as_matrix()
    state_weights = tf.convert_to_tensor(state_weights, dtype=tf.float32)
    ev_states = ev_states[state_polls.state.unique()].as_matrix()

    n_states = len(state_polls.state.unique())
    n_pollsters = len(polls.pollster.unique())
    # Day and week of last poll
    t_last = state_polls.date_index.max()
    w_last = state_polls.week_index.max()
    # Day of beginning of last week
    last_tuesday = t_last - state_polls.day_of_week[state_polls.date_index.argmax()]
    # Days until election
    days_until_E = (ELECTION_DATE - state_polls.poll_date.max().date()).days
    # Election day as index
    E_day = days_until_E + state_polls.date_index.max()
    # weeks_until_E = math.floor(days_until_E / 7)
    E_week = (ELECTION_DATE + dt.timedelta(days=-1)).isocalendar()[1] - polls.week.min()

    # MODEL
    # FORWARD COMPONENT
    # Forecast priors - for dates from t_last to election day
    # Latent State vote intention
    mu_b_prior_cov = tf.convert_to_tensor(covariance_matrix(0.05, 0.5, n_states), dtype=tf.float32)
    mu_b_prior_mean = tf.convert_to_tensor(logit(0.486 + prior_diff_score).as_matrix(), dtype=tf.float32)
    mu_b_prior = MultivariateNormalFullCovariance(loc=mu_b_prior_mean, covariance_matrix=mu_b_prior_cov)

    # Reverse random walk for dates where we don't have polls
    mu_bs = []
    mu_bs.append(mu_b_prior)
    sigma_walk_b_forecast = covariance_matrix(7 * 0.015 ** 2, 0.75, n_states)
    sigma_walk_b_forecast = tf.convert_to_tensor(sigma_walk_b_forecast, dtype=tf.float32)
    for w in range(E_week - state_polls.week_index.max()):
        mu_bs.append(MultivariateNormalFullCovariance(loc=mu_bs[-1], covariance_matrix=sigma_walk_b_forecast))

    # BACKWARD COMPONENT
    # Backward priors - from t_last to first day of polling
    # Latent State vote intention
    sigma_b = Normal(loc=-8.0, scale=0.5)
    for w in range(w_last):
        mu_bs.append(NormalWithSoftplusScale(loc=mu_bs[-1], scale=sigma_b))

    # Latent national component
    sigma_a = Normal(loc=-3.0, scale=1.0)
    mu_a_buffer = tf.zeros(1, tf.float32)
    mu_as = []
    for t in range(E_day):
        if t == 0:
            mu_as.append(NormalWithSoftplusScale(loc=0.0, scale=sigma_a))
        else:
            mu_as.append(NormalWithSoftplusScale(loc=mu_as[-1], scale=sigma_a))

    # Pollster house effect
    sigma_c = Normal(loc=-4.0, scale=0.5)
    mu_c = NormalWithSoftplusScale(loc=tf.zeros(n_pollsters), scale=sigma_c)

    # Sampling error
    # samp_e_state = Normal(loc=tf.zeros(len(state_polls)), scale=0.13)
    # samp_e_state = tf.random_normal([len(state_polls)], mean=0.0, stddev=0.13)

    # State polling error
    # sigma_poll_error = covariance_matrix(0.02, 0.75, n_states)
    # sigma_poll_error = tf.convert_to_tensor(sigma_poll_error, dtype=tf.float32)
    # e = MultivariateNormalFullCovariance(loc=tf.zeros(n_states), covariance_matrix=sigma_poll_error)

    # STATE POLLS
    mu_b_tf = tf.stack(mu_bs)
    mu_a_tf = tf.stack(mu_as)
    mu_a_tf = tf.concat([mu_a_buffer, mu_a_tf], axis=0)
    # # Due to list in reverse
    mu_a_state = tf.gather(mu_a_tf, (E_day - state_polls.date_index).as_matrix())
    state_ind = state_polls[['week_index', 'state_index']].as_matrix()
    # Due to list in reverse
    state_ind[:, 0] = E_week - state_ind[:, 0]

    mu_b_state = tf.gather_nd(mu_b_tf, state_ind)
    mu_c_state = tf.gather(mu_c, state_polls.pollster_index)
    # e_state = tf.gather(e, state_polls.state_index.as_matrix())

    state_logits = mu_b_state + mu_a_state #+ e_state

    # NATIONAL POLLS
    nat_ind = national_polls[['week_index', 'date_index']].as_matrix()
    # Due to list in reverse
    nat_ind[:, 0] = E_week - nat_ind[:, 0]
    nat_ind[:, 1] = E_day - nat_ind[:, 1]
    mu_b_nat = tf.gather(mu_b_tf, nat_ind[:, 0])
    mu_a_nat = tf.expand_dims(tf.gather(mu_a_tf, nat_ind[:, 1]), 1)
    # expit
    nat_expits = 1 / (1 + tf.exp(-1 * (mu_a_nat + mu_b_nat)))# + e)))
    # logit
    nat_weigh_avg = -tf.log((1 / (tf.reduce_sum(tf.multiply(state_weights, nat_expits), axis=1))) - 1)
    mu_c_nat = tf.gather(mu_c, national_polls.pollster_index)
    # alpha = Normal(loc=logit())

    final_logits = tf.concat([state_logits, nat_weigh_avg], axis=0)
    final_logits += tf.concat([mu_c_state, mu_c_nat], axis=0)

    X = tf.placeholder(tf.float32, len(state_polls) + len(national_polls))
    y = Binomial(total_count=X, logits=final_logits, value=tf.zeros(len(state_polls) + len(national_polls), dtype=tf.float32))

    # INFERENCE
    sigmas = [sigma_a, sigma_b, sigma_c]
    others = [mu_c]
    latent_variables = mu_bs + mu_as + others + sigmas
    n_respondents = np.append(state_polls.n_respondents.as_matrix(), national_polls.n_respondents.as_matrix())
    n_clinton = np.append(state_polls.n_clinton.as_matrix(), national_polls.n_clinton.as_matrix())
    # 10,000 samples default
    inference = ed.HMC(latent_variables, data={X: n_respondents, y: n_clinton})
    inference.initialize(n_print=100, step_size=0.006, n_steps=2)

    tf.global_variables_initializer().run()
    for t in range(inference.n_iter):
        info_dict = inference.update()
        inference.print_progress(info_dict)

        if t % inference.n_print == 0:
            print(inference.latent_vars[latent_variables[0]].params.eval()[t])
            print(inference.latent_vars[latent_variables[23]].params.eval()[t])
            print(inference.latent_vars[mu_c].params.eval()[t])

    # Extract samples
    qmu_bs = []
    for b in mu_bs:
        qmu_bs.append(inference.latent_vars[b].params.eval())
    qmu_bs = list(reversed(qmu_bs))

    qmu_as = []
    for a in mu_as:
        qmu_as.append(inference.latent_vars[a].params.eval())
    qmu_as = list(reversed(qmu_as))

    qmu_c = inference.latent_vars[mu_c].params.eval()

    date_index = state_polls.date_index.as_matrix()
    week_index = state_polls.week_index.as_matrix()
    predicted_scores = predict_scores(qmu_as, qmu_bs, date_index, week_index, last_tuesday, E_day)

    i = 0
    for s in state_polls.state.unique():
        state_s_polls = state_polls[state_polls.state == s]
        state_scores = predicted_scores[:, :, i]
        # generate_plot(state_scores, this_state_polls, burn_in=4000)
        i += 1

    # Apply burn in
    predicted_scores = predicted_scores[:, BURN_IN:, :]

    # SIMULATE ELECTION
    e_day_results = predicted_scores[-1, :, :]
    outcomes = []
    for i in range(10000):
        draw = np.random.randint(0, e_day_results.shape[1])
        outcome = e_day_results[draw]
        outcome = np.dot(outcome >= 0.5, ev_states)
        outcomes.append(outcome)
    x = np.unique(outcomes)
    freq = collections.Counter(outcomes)
    height = [freq[s] for s in x]
    plt.bar(x, height)
    plt.show()


    week = 0
    election_day = inference.latent_vars[latent_variables[week]].params.eval()
    # Burn in
    election_day = election_day[BURN_IN:]
    # election_day = np.unique(election_day, axis=0)
    print(np.mean(election_day, axis=0))
    print(np.std(election_day, axis=0))
    # election_day = np.unique(election_day, axis=0)

    week =27
    first_week = inference.latent_vars[latent_variables[week]].params.eval()
    # Burn in
    first_week = first_week[BURN_IN:]
    print(np.mean(first_week, axis=0))
    print(np.std(first_week, axis=0))
    # first_week = np.unique(first_week, axis=0)

    week =-4
    house_effects = inference.latent_vars[latent_variables[week]].params.eval()
    # Burn in
    house_effects = house_effects[BURN_IN:]
    print(np.mean(house_effects, axis=0))
    print(np.std(house_effects, axis=0))
    # house_effects = np.unique(house_effects, axis=0)

    predicted_scores[:, :, -2][:, BURN_IN:][:, 1]
