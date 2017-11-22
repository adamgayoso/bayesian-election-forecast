import pollster
import pandas as pd
import datetime as dt

START_DATE = dt.date(2016, 5, 1)
ELECTION_DATE = dt.date(2016, 11, 8)
end_data = ELECTION_DATE


def filter_polls(all_polls, end_date, undecided=False):
    """Filter the raw all_polls into clean_polls for Edward.
        Code modified from:
        https://github.com/fonnesbeck/election_pycast/blob/master/Election2016.ipynb

    Args:
        all_polls (DataFrame): raw all_polls DF from HuffPo api
        end_date (dt.date): Date for which polls will be considered
        undecided (bool, optional): Keep undecided voters in clean_polls

    Returns:
        pd.DataFrame: Filtered clean_polls
    """
    clean_polls = all_polls.copy()

    clean_polls['end'] = clean_polls['end_date']
    clean_polls['begin'] = clean_polls['start_date']
    clean_polls['poll_time'] = (clean_polls.end - clean_polls.begin).dt.days
    poll_date = clean_polls.end - ((clean_polls.end - clean_polls.begin) / 2)
    clean_polls['poll_date'] = poll_date.dt.date
    # Day 0 for any week should be Tuesday (default is Monday)
    clean_polls['week'] = (poll_date + dt.timedelta(days=-1)).dt.week
    clean_polls['day_of_week'] = (poll_date + dt.timedelta(days=-1)).dt.weekday
    clean_polls['pollster'] = clean_polls.survey_house

    # Consolidate third-party candidates into other
    other = clean_polls[['johnson', 'mcmullin', 'other']].fillna(0).sum(axis=1)
    clean_polls['other'] = other
    # This is a percentage * 100
    clean_polls['both'] = clean_polls.clinton + clean_polls.trump
    clean_polls.undecided = clean_polls.undecided.fillna(0)
    clean_polls['p_undecided'] = 100 * clean_polls.undecided / \
        (clean_polls.both + clean_polls.undecided)

    rows = (clean_polls.observations > 1)
    rows = rows & (clean_polls.poll_date >= START_DATE)
    rows = rows & (clean_polls.end_date <= end_date)

    cols = ['state', 'begin', 'end', 'poll_time', 'poll_date',
            'week', 'day_of_week', 'pollster', 'mode', 'population',
            'observations', 'clinton', 'trump', 'both',
            'other', 'undecided', 'p_undecided']

    clean_polls = clean_polls.loc[rows, cols]
    clean_polls['p_clinton'] = clean_polls.clinton / clean_polls.both
    clean_polls['n_clinton'] = round(
        clean_polls.observations * clean_polls.clinton / 100)
    clean_polls['n_respondents'] = round(
        clean_polls.observations * clean_polls.both / 100)

    # Questionable. Do we need to remove these? There are 40 out of 1437
    clean_polls = clean_polls.drop_duplicates(
        ['state', 'poll_date', 'pollster'])

    return clean_polls


def get_polls(end_date):
    """Get polling data from HuffPo's Pollster API

    Args:
        end_date (dt.date): Date until which polls would like to be considered

    Returns:
        pd.DataFrame: Filtered clean_polls
    """
    api = pollster.Api()
    charts = api.charts_get(
        cursor=None,
        tags='2016-president',
        election_date=dt.date(2016, 11, 8)
    )

    polls = []
    two_words = ['west', 'north', 'south', 'new', 'rhode']
    for item in charts.items:

        if item.slug == '2016-general-election-trump-vs-clinton-vs-johnson':
            slug = '2016-general-election-trump-vs-clinton'
        else:
            slug = item.slug
        chart = api.charts_slug_pollster_chart_poll_questions_tsv_get(slug)
        if slug.split('-')[1] in two_words:
            state = slug.split('-')[1] + ' ' + slug.split('-')[2]
        elif slug == '2016-washington-d-c-president-trump-vs-clinton':
            state = 'district of columbia'
        else:
            state = slug.split('-')[1]
        chart = chart.assign(state=state)
        polls.append(chart)

    other = ['2016-california-presidential-general-election-trump-vs-clinton',
             '2016-florida-presidential-general-election-trump-vs-clinton']
    for slug in other:
        chart = api.charts_slug_pollster_chart_poll_questions_tsv_get(slug)
        chart = chart.assign(state=slug.split('-')[1])
        polls.append(chart)

    all_polls = pd.concat(polls, axis=0)
    all_polls.columns = all_polls.columns.str.lower()
    clean_polls = filter_polls(all_polls, end_date)

    return clean_polls


if __name__ == '__main__':

    polls = get_polls(ELECTION_DATE)
    polls.to_csv('data/all_polls_2016.csv')
