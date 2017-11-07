import pandas as pd
import numpy as np
import edward as ed
import tensorflow as tf
from edward.models import Normal, Binomial, MultivariateNormalFullCovariance, Uniform, Empirical, InverseGamma, Exponential
from scipy.special import logit
import datetime as dt
# import math
from scipy.stats import binom
from helper import _sample_n, prepare_polls, process_2012_polls

ELECTION_DATE = dt.date(2016, 11, 8)
pd.options.mode.chained_assignment = None

def main():

    # Load data
    polls = pd.read_csv('data/all_polls_2016.csv', parse_dates=['begin', 'end', 'poll_date'])
    up_to_t = dt.date(2016, 11, 8)
    state_polls, national_polls = prepare_polls(polls, up_to_t)

    # Get prior information from 2012 election
    prior_diff_score, state_weights, ev_states = process_2012_polls()
    prior_diff_score = prior_diff_score[state_polls.state.unique()]

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

    # TODO make a covariance matrix function
    # FORWARD COMPONENT
    # Forecast priors - for dates from t_last to election day
    # Latent State vote intention
    mu_b_prior_cov = tf.convert_to_tensor(0.025 * np.ones((n_states, n_states)) + 0.025 * np.identity(n_states), dtype=tf.float32)
    mu_b_prior_mean = tf.convert_to_tensor(logit(0.486 + prior_diff_score).as_matrix(), dtype=tf.float32)
    mu_b_prior = MultivariateNormalFullCovariance(loc=mu_b_prior_mean, covariance_matrix=mu_b_prior_cov)

    # Reverse random walk for dates where we don't have polls
    mu_bs = []
    mu_bs.append(mu_b_prior)
    sigma_walk_b_forecast = 0.00118 * np.ones((n_states, n_states)) + ((7 * 0.015 ** 2) - 0.00118) * np.identity(n_states)
    sigma_walk_b_forecast = tf.convert_to_tensor(sigma_walk_b_forecast, dtype=tf.float32)
    for w in range(E_week - state_polls.week_index.max()):
        mu_bs.append(MultivariateNormalFullCovariance(loc=mu_bs[-1], covariance_matrix=sigma_walk_b_forecast))

    # BACKWARD COMPONENT
    # Backward priors - from t_last to first day of polling
    # Latent State vote intention
    sigma_b = 0.01 * tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))
    # sigma_b = Exponential(rate=1.0)
    # constrained_sigma_b = 0.05 * tf.exp(-sigma_b) * np.sqrt(7)
    for w in range(w_last):
        mu_bs.append(Normal(loc=mu_bs[-1], scale=sigma_b * np.sqrt(7) * tf.ones(n_states)))

    # Latent national component
    sigma_a = 0.1 * tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))
    # sigma_a = Exponential(rate=1.0)
    # constrained_sigma_a = 0.05 * tf.exp(-sigma_a)

    # mu_a_base = Normal(loc=tf.zeros(last_tuesday+1), scale=0.025 * tf.ones(last_tuesday+1))
    # mu_as = tf.cumsum(mu_a_base)

    # How can we vectorize this?
    mu_a_buffer = tf.zeros(t_last - last_tuesday + 1, tf.float32)
    mu_as = []
    for t in range(last_tuesday):
        if t == 0:
            mu_as.append(Normal(loc=0.0, scale=sigma_a))
        else:
            mu_as.append(Normal(loc=mu_as[-1], scale=sigma_a))

    # Pollster house effect
    # sigma_c = InverseGamma(2.0, 0.04)
    # # sigma_c = Exponential(rate=1.0)
    # # constrained_sigma_c = 0.1 * tf.exp(-sigma_c)
    # mu_c = Normal(loc=tf.zeros(n_pollsters), scale=sigma_c * tf.ones(n_pollsters))

    # # # Sampling error
    # sigma_samp_e_state = InverseGamma(2.0, 0.04)
    # sigma_samp_e_state = Exponential(rate=1.0)
    # constrained_same_e_state =0.1 * tf.exp(-sigma_samp_e_state)
    # sigma_samp_e_nat = Uniform(low=0.0, high=0.1)
    # samp_e_state = Normal(loc=tf.zeros(len(state_polls)), scale=sigma_samp_e_state * tf.ones(len(state_polls)))

    # State polling error
    # sigma_poll_error = 0.00112 * np.ones((n_states, n_states)) + ((0.0016) - 0.00112) * np.identity(n_states)
    # sigma_poll_error = tf.convert_to_tensor(sigma_poll_error, dtype=tf.float32)
    # e = MultivariateNormalFullCovariance(loc=tf.zeros(n_states), covariance_matrix=sigma_poll_error)

    # Binomial logits using tf gather to get the right values.
    mu_b_tf = tf.stack(mu_bs)
    mu_a_tf = tf.stack(mu_as)
    mu_a_tf = tf.concat([mu_a_buffer, mu_a_tf], axis=0)
    # # Due to list in reverse
    mu_a_log = tf.gather(mu_a_tf, (t_last - state_polls.date_index).as_matrix())
    ind = state_polls[['week_index', 'state_index']].as_matrix()
    # Due to list in reverse
    ind[:, 0] = E_week - ind[:,0]

    mu_b_log = tf.gather_nd(mu_b_tf, ind)
    # mu_c_log = tf.gather(mu_c, state_polls.pollster_index)
    # e_log = tf.gather(e, state_polls.state_index)

    log_lin = mu_b_log + mu_a_log# + e_log + samp_e_state # + mu_c_log
    Binomial._sample_n = _sample_n
    X = tf.placeholder(tf.float32, len(state_polls))
    y = Binomial(total_count=X, logits=log_lin)#, value=tf.zeros(len(state_polls), dtype=tf.float32))

    # Inference
    # sigmas = [sigma_a, sigma_b]#, sigma_samp_e_state]
    # others = [samp_e_state]
    latent_variables = mu_bs + mu_as# + others
    # Feeding a list does 10000 iter by default
    n_respondents = state_polls.n_respondents.as_matrix()
    n_clinton = state_polls.n_clinton.as_matrix()
    inference = ed.HMC(latent_variables, data={X: n_respondents, y: n_clinton})
    inference.initialize(n_print=10, step_size=0.0028, n_steps=2)

    tf.global_variables_initializer().run()
    for t in range(inference.n_iter):
        info_dict = inference.update()
        inference.print_progress(info_dict)

        # if t % inference.n_print == 0:
        #     print(inference.latent_vars[latent_variables[-1]].sample().eval())
    week = 0
    election_day = inference.latent_vars[latent_variables[week]].params.eval()
    # Burn in
    election_day = election_day[1000:]
    print(np.mean(election_day, axis=0))
    print(np.std(election_day, axis=0))
    # election_day = np.unique(election_day, axis=0)

    week = 32
    first_week = inference.latent_vars[latent_variables[week]].params.eval()
    # Burn in
    first_week = first_week[1000:]
    print(np.mean(first_week, axis=0))
    print(np.std(first_week, axis=0))

    latents = list(inference.latent_vars.keys())
    vari = inference.latent_vars[latents[-1]].params.eval()
    vari = vari[1000:]
    np.mean(vari)
    # vari = np.unique(vari)

    house_effects = inference.latent_vars[latent_variables[-2]].params.eval()
    house_effects = house_effects[1000:]
    np.mean(house_effects, axis=0)

    mu_a = inference.latent_vars[latent_variables[-1]].params.eval()
    mu_a = mu_a[1000:]
    np.mean(mu_a, axis=0)


