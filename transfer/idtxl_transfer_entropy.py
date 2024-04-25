from __future__ import annotations

import collections
import datetime as dt
import itertools
import random
import pickle
import datetime as dt

import numpy as np
import pandas as pd

import idtxl
from idtxl.data import Data
from idtxl.bivariate_te import BivariateTE

def load_and_process_topics(url_to_ts, url_to_topic_scores, n_topics=10):
    """
    Given a dict (pkl) and a mapping from url to topic scores, produce a pandas dataframe w/ columns [ts, topic1_count, topic2_count,...]

    Inputs:
        url_to_ts: dict of {url: publication ts}
        url_to_topic_scores: dict of {url: [topic1_score, topic2_score, ...]
        n_topics: number of topics; default 10

    Outputs:
        df_ts_topic: pandas dataframe w/ columns [ts, topic1_count, topic2_count,...]
    """
    ts_to_topics = {}
    for url, ts in url_to_ts.items():
        kw_scores = url_to_topic_scores[url]
        ts_round = pd.to_datetime(ts).floor('D')
        if ts_round in ts_to_topics:
            for idx, sc in enumerate(kw_scores):
                ts_to_topics[ts_round][idx] += sc
        else:
            ts_to_topics[ts_round] = {}
            for idx, sc in enumerate(kw_scores):
                ts_to_topics[ts_round][idx] = sc

    def update_ts_kw_dict(my_ts):
        """
        Given a timestamp, make a dict that contains all the keyword counts for all valid keywords

        Input: my_ts: pandas timestamp

        Output: d {keyword: count}
        """
        d = {'ts': my_ts}
        ts_topic_dict = ts_to_topics.get(my_ts, {})
        for kw in range(n_topics):
            d[topic] = ts_topics_dict.get(topic, 0)
        return d

    ts_topic_dicts = []
    for ts in list(pd.date_range(start='2023-04-01', end='2023-06-30', freq='D')):
        vals = ts_to_topics.get(ts, {})
        d = {'ts': ts}
        for i in range(n_topics):
            d[i] = vals.get(i, 0)
        ts_topic_dicts.append(d)

    return pd.DataFrame(ts_topic_dicts)


def get_media_outlet(media_group, url):
    if media_group in STATES:
        return media_group
    else:
        if 'nytimes.com' in url:
            return 'nytimes.com'
        elif 'foxnews.com' in url:
            return 'foxnews.com'
        else:
            return None


te_results = {i: {} for i in range(20)}
STATES = set(['newyork', 'florida', 'california', 'ohio', 'illinois', 'texas'])
N_TOPICS = 20
combined_df = pd.read_csv('/home/pranavgoel/trans-fer-entropy/topic_modeling/apr_jun_2023_post_irrelevant_article_filtering/all_texts_combined_post_irrelevant_article_filtering.csv') 
combined_df['publish_date'] = combined_df['publish_date'].apply(pd.to_datetime)
# combined_df = combined_df[combined_df['publish_date'] > dt.datetime(year=2023, month=5, day=8)]
print([c for c in combined_df.columns])
combined_df['publication'] = combined_df.apply(lambda b: get_media_outlet(b.media_group, b.url), axis=1)


# for state in ['newyork', 'florida', 'california', 'ohio', 'illinois', 'texas']:
for state in ['nytimes.com', 'foxnews.com']:
    df_x = combined_df[combined_df['publication'] == state]
    for pub in ['nytimes.com', 'foxnews.com']:
    # for pub in ['newyork', 'florida', 'california', 'ohio', 'illinois', 'texas']:
        df_y = combined_df[combined_df['publication'] == pub]

        print([c for c in df_x.columns])
        print([c for c in df_y.columns])
        pkl_x = {url: pd.to_datetime(ts) for url, ts in zip(df_x['url'], df_x['publish_date'])}
        pkl_y = {url: pd.to_datetime(ts) for url, ts in zip(df_y['url'], df_y['publish_date'])}
        url_to_topic_scores = pickle.load(open('/home/pranavgoel/trans-fer-entropy/topic_modeling/apr_jun_2023_post_irrelevant_article_filtering/url_to_topic_distribution_for_num_topics_{}.pkl'.format(str(N_TOPICS)), 'rb'))
        df_x_topics = load_and_process_topics(pkl_x, url_to_topic_scores, N_TOPICS)
        df_y_topics = load_and_process_topics(pkl_y, url_to_topic_scores, N_TOPICS)
        
        # overlap_kws = kw_x.intersection(kw_y)
        
        kw_scores_y_on_x = []
        kw_scores_x_on_y = []
        st_ls = [state, pub]
        for kw in range(N_TOPICS):
            x = np.array(df_x_topics[kw])
            y = np.array(df_y_topics[kw])
            arr = np.vstack([x, y])
            dat = Data(arr, dim_order='ps')
            network_analysis = BivariateTE()
            settings = {'cmi_estimator': 'JidtGaussianCMI',
                        'max_lag_sources': 5,
                        'min_lag_sources': 1}
            results = network_analysis.analyse_network(settings=settings, data=dat)
            try:
                te_res = results.get_adjacency_matrix(weights='max_te_lag', fdr=False).get_edge_list()
                for res in te_res:
                    tup = (st_ls[res[0]], st_ls[res[1]])
                    te_results[kw][tup] = res[2]
            except Exception as e:
                print(e)
                continue

pickle.dump(te_results, open('idtxl_all_fox_filtered_te_results_20_topics_pubs_and_pubs', 'wb'))
