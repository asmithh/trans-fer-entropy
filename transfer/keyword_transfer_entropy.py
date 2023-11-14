from __future__ import annotations

import collections
import datetime as dt
import itertools
import pickle

import pandas as pd

from symbolic_transfer_entropy import symbolic_transfer_entropy

def load_and_process_pickle(pkl, count_threshold=5):
    pkl_keywords = list(itertools.chain(*[[vv[1] for vv in v[1]] for v in pkl.values()]))
    pkl_kw_counter = collections.Counter(pkl_keywords)
    thresholded_kws = set([k  for k, v in pkl_kw_counter.items() if v >= count_threshold])

    ts_to_keywords = {}
    ts_to_url_to_keywords = {}
    for url, kw_list in pkl.items():
        ts = kw_list[0]
        scores = kw_list[1]
        keep_kws = [s[1] for s in scores if s[1] in thresholded_kws]
        ts_round = pd.to_datetime(ts).floor('D')
        if ts_round in ts_to_keywords:
            ts_to_keywords[ts_round] += keep_kws
        else:
            ts_to_keywords[ts_round] = keep_kws

        if ts_round in ts_to_url_to_keywords:
            if url in ts_to_url_to_keywords[ts_round]:
                ts_to_url_to_keywords[ts_round][url] += keep_kws
            else:
                ts_to_url_to_keywords[ts_round][url] = keep_kws
        else:
            ts_to_url_to_keywords[ts_round] = {url: keep_kws}


    ts_to_keywords = {ts: collections.Counter(kws) for ts, kws in ts_to_keywords.items()}
    ts_to_url_to_keywords = {ts: {url: collections.Counter(kws) for url, kws in val.items()} for ts, val in ts_to_url_to_keywords.items()}

    def update_ts_kw_dict(my_ts):
        d = {'ts': my_ts}
        ts_kw_dict = ts_to_keywords.get(my_ts, {})
        for kw in list(thresholded_kws):
            d[kw] = ts_kw_dict.get(kw, 0)
        return d

    df_ts_kw = [update_ts_kw_dict(ts) for ts in list(pd.date_range(start='2023-04-01', end='2023-06-30', freq='D'))]
    df_ts_kw = pd.DataFrame(df_ts_kw)

    return df_ts_kw, thresholded_kws
    
pkl_x = pickle.load(open('/home/pranavgoel/trans-fer-entropy/keyword_extraction/california_url_to_dated_keywords.pkl', 'rb'))
pkl_y = pickle.load(open('/home/pranavgoel/trans-fer-entropy/keyword_extraction/nytimes_foxnews_url_to_dated_keywords.pkl', 'rb'))

pkl_y_nyt = {k: v for k, v in pkl_y.items() if 'nytimes.com' in k}
df_x, kw_x = load_and_process_pickle(pkl_x)
df_y, kw_y = load_and_process_pickle(pkl_y_nyt)

print(df_x.head(10))
overlap_kws = kw_x.intersection(kw_y)

kw_scores_y_on_x = []
kw_scores_x_on_y = []
for kw in list(overlap_kws):
    print(kw)
    x = list(df_x[kw])
    y = list(df_y[kw])
    kw_scores_y_on_x.append((kw, symbolic_transfer_entropy(x, y, 5)))
    kw_scores_x_on_y.append((kw, symbolic_transfer_entropy(y, x, 5)))

with open('nyt_on_ca_w_5.txt', 'w') as f:
    for line in sorted(kw_scores_y_on_x, key=lambda b: b[1], reverse=True)[0:200]:
        f.write(','.join([str(l) for l in line]) + '\n')

with open('ca_on_nyt_w_5.txt', 'w') as f:
    for line in sorted(kw_scores_x_on_y, key=lambda b: b[1], reverse=True)[0:200]:
        f.write(','.join([str(l) for l in line]) + '\n')
