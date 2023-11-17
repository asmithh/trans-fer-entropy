"""
This code borrows from extract_and_save_keywords_for_all_articles.ipynb, and creates separate keyword-timeseries for nytimes and foxnews (as opposed to one "national outlet" that contains both those outlets as done previously).

@author: Pranav Goel
"""

import os
import pickle

import numpy as np
import pandas as pd
from rake_nltk import Rake
from tqdm.auto import tqdm


INPUT_NYTIMES_FOXNEWS_FILEPATH = "../obtaining_news_collections/data/nytimes_foxnews_article_texts_and_info_dedup.csv"


def get_dated_keywords(ts_text_tuples, min_length=1, max_length=3):
    """
    Given ts_text_tuples, a list of (timestamp, article_text) tuples,
    extract & return a list of timestamped keywords + scores using rake-nltk.
    """
    dated_keywords = []
    r = Rake(min_length=min_length, max_length=max_length)
    for tup in tqdm(ts_text_tuples):
        r.extract_keywords_from_text(tup[1])
        kw = r.get_ranked_phrases_with_scores()
        dated_keywords.append((tup[0], kw))

    return dated_keywords


def get_dates_and_title_texts_lists(df):
    """
    Uses dates, titles, and story texts in the pandas dataframe (df) to return a list of a list of (timestamp, article_text) tuples where article_text combines the title and story text.
    """
    # urls = list(df['url'])
    dates = list(df["publish_date"])
    titles_and_texts = list(zip(df["title"], df["text"]))
    titles_and_texts = list(map(lambda x: x[0] + "\n\n\n" + x[1], titles_and_texts))
    return list(zip(dates, titles_and_texts))


if __name__ == "__main__":
    df = pd.read_csv(INPUT_NYTIMES_FOXNEWS_FILEPATH)
    nytimes_df = df[df["media_name"] == "New York Times"]
    foxnews_df = df[df["media_name"] == "Fox News"]
    del df

    nytimes_l = get_dates_and_title_texts_lists(nytimes_df)
    foxnews_l = get_dates_and_title_texts_lists(foxnews_df)

    nytimes_dated_keywords = get_dated_keywords(nytimes_l)
    foxnews_dated_keywords = get_dated_keywords(foxnews_l)

    # save
    pickle.dump(
        dict(zip(list(nytimes_df["url"]), nytimes_dated_keywords)),
        open("nytimes_url_to_dated_keywords.pkl", "wb"),
    )

    pickle.dump(
        dict(zip(list(foxnews_df["url"]), foxnews_dated_keywords)),
        open("foxnews_url_to_dated_keywords.pkl", "wb"),
    )
