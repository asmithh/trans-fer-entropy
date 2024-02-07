"""
This code borrows trains MALLET topic model(s) on combined article text data, saving all the outputs of the model, and also adding the topic probabilities to the combined data file. And saving the url to topic distribution mapping itself. Topics themselves will be interpreted and displayed (in terms of top words and top documents) in interpret_topic_modeling_output.ipynb.

Attributes:
    DOMAIN_CLASSIFICATIONS_FILEPATH: File with domain classifications (local, national, INCONSISTENT); see README for details. 
    INPUT_DEDUP_FILES_BASE_PATH: Directory path to the deduplicated data files of all media groups. 
    INPUT_FILE_END_PATTERN: Input data files start with the state name and end with this string. 
    STATE_NAMES: Names of state used in state collections which also form the beginning of input files. 
    OUTPUT_PATH: Directory path to save csv files with state collections having national outlet articles removed (also copied over nytimes_foxnews file to have all data in one place). 
    
Functions:
    obtain_national_domain_set: Return set of national news domains using existing classification dataset. 
    topic_model_training: Using the full dataframe, obtains the texts (article headlines and stories), preprocesses them to create training data, and trains various topic model for different values of K or num_topics. 
    add_topic_distribution_to_combined_data_csv: For a specified value of K (num_topics), loads the doc-topic distributions, and saves that as added columns to the csv data. 
    save_url_to_topic_distributions_mapping: For a specified value of K (num_topics), loads the doc-topic distributions, and saves that as URL -> list of topic distribution values.

@author: Pranav Goel
"""

import os
import pickle

import numpy as np
import pandas as pd
import tldextract

DOMAIN_CLASSIFICATIONS_FILEPATH = "/home/pranavgoel/trans-fer-entropy/national_outlet_filtering_in_state_collections/domain_classification_data/combined_clean_local_national_domain_classifications.csv"

INPUT_DEDUP_FILES_BASE_PATH = "/home/pranavgoel/trans-fer-entropy/obtaining_news_collections/data/prior_data_apr_jun_2023/"

INPUT_FILE_END_PATTERN = "_article_texts_and_info_dedup.csv"

STATE_NAMES = ["california", "texas", "illinois", "ohio", "florida", "newyork"]

OUTPUT_PATH = "/home/pranavgoel/trans-fer-entropy/national_outlet_filtering_in_state_collections/data_with_national_outlets_removed_in_state_collections/"


def obtain_national_domain_set():
    """
    Return set of national news domains using existing classification dataset.

    OUTPUT:
        national_domains: Set of national news domains
    """
    domain_classification_df = pd.read_csv(DOMAIN_CLASSIFICATIONS_FILEPATH)
    national_domains = set(
        domain_classification_df[
            domain_classification_df["classification"] == "national"
        ]["domain"]
    )
    print("Number of national news domains = " + str(len(national_domains)))
    print("\n\n")
    return national_domains


def filter_national_domains_in_state_collections_and_display_stats(national_domains):
    """
    Filter out national domain in state article collection files, display pre- and post-filtering sizes for each state, and save the new files.

    INPUT:
        national_domains: Set of national news domains
    """
    for state in STATE_NAMES:
        df = pd.read_csv(INPUT_DEDUP_FILES_BASE_PATH + state + INPUT_FILE_END_PATTERN)
        urls = list(df["url"])
        domains = [tldextract.extract(u).registered_domain for u in urls]
        df["domain"] = domains
        print("For state: " + state)
        print("#Articles before national-outlet-filter = " + str(len(df)))
        df = df[~df["domain"].isin(national_domains)]
        print("#Articles after national-outlet-filter = " + str(len(df)))
        df.to_csv(
            OUTPUT_PATH
            + state
            + "_article_texts_and_info_dedup_without_national_outlets.csv"
        )
        print("\n========\n")


if __name__ == "__main__":
    national_domains = obtain_national_domain_set()

    filter_national_domains_in_state_collections_and_display_stats(national_domains)
