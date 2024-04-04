"""
This code borrows trains MALLET topic model(s) on combined article text data -- AFTER FILTERING OUT NATIONAL OUTLETS IN STATE MEDIA + FILTERING OUT IRRELEVANT ARTICLES, saving all the outputs of the model, and also adding the topic probabilities to the combined data file. And saving the url to topic distribution mapping itself. Topics themselves will be interpreted and displayed (in terms of top words and top documents) in a separate notebook. 

Attributes:
    INPUT_DEDUP_FILES_BASE_PATH: Directory path to the deduplicated data files of all media groups. 
    OUTPUT_COMBINED_DATAFILE_PATH: File path to save csv file containing all the articles across all media groups. 
    MALLET_PATH: Path to mallet exe script needed for topic modeling functionality (clone: https://github.com/mimno/Mallet and built using ant)
    NUM_TOPIC_VALUES: List of values of K or number of topics to use for running mallet. 
    MALLET_OUTPUT_PATH: Path to contain all the various outputs created by mallet during topic modeling training. 
    OUTPUT_DATA_WITH_TOPIC_DISTRIBUTION_BASEPATH: Path to save the various files that include doc-topic distributions for articles. 

Functions:
    create_and_save_combined_dataframe: Loads all the dedup files representing various media groups (the 6 states + nytimes+foxnews) and concatenates and saves them as one dataframe, while adding a column for the media group. 
    topic_model_training: Using the full dataframe, obtains the texts (article headlines and stories), preprocesses them to create training data, and trains various topic model for different values of K or num_topics. 
    add_topic_distribution_to_combined_data_csv: For a specified value of K (num_topics), loads the doc-topic distributions, and saves that as added columns to the csv data. 
    save_url_to_topic_distributions_mapping: For a specified value of K (num_topics), loads the doc-topic distributions, and saves that as URL -> list of topic distribution values.

@author: Pranav Goel
"""

import os
import pickle

import little_mallet_wrapper as lmw  # https://github.com/maria-antoniak/little-mallet-wrapper/tree/master
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

INPUT_DEDUP_FILES_BASE_PATH = "/home/pranavgoel/trans-fer-entropy/article_filtering/transgender_filtered_data_files/"

OUTPUT_COMBINED_DATAFILE_PATH = "/home/pranavgoel/trans-fer-entropy/topic_modeling/apr_jun_2023_post_irrelevant_article_filtering/all_texts_combined_post_irrelevant_article_filtering.csv"

MALLET_PATH = "/home/pranavgoel/Mallet/bin/mallet"

NUM_TOPIC_VALUES = [20]

MALLET_OUTPUT_PATH = "/home/pranavgoel/mallet_output/trans_fer_entropy_apr_jun_2023_post_irrelevant_article_filtering"

OUTPUT_DATA_WITH_TOPIC_DISTRIBUTION_BASEPATH = "/home/pranavgoel/trans-fer-entropy/topic_modeling/apr_jun_2023_post_irrelevant_article_filtering/"


def create_and_save_combined_dataframe(input_file_names):
    """
    Loads all the dedup files representing various media groups (the 6 states + nytimes+foxnews) and concatenates and saves them as one dataframe, while adding a column for the media group.
    """
    all_dfs = []
    for f in input_file_names:
        df = pd.read_csv(INPUT_DEDUP_FILES_BASE_PATH + f)
        media_group = f.split("_article_")[0]
        df["media_group"] = [media_group] * len(df)
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs)
    combined_df.to_csv(OUTPUT_COMBINED_DATAFILE_PATH, index=None)
    return combined_df


def topic_model_training(df):
    """
    Using the full dataframe, obtains the texts (article headlines and stories), preprocesses them to create training data, and trains various topic model for different values of K or num_topics.
    """
    titles_and_texts = list(zip(df["title"], df["text"]))
    titles_and_texts = list(map(lambda x: x[0] + "\n\n\n" + x[1], titles_and_texts))
    training_data = [
        lmw.process_string(t, numbers="remove") for t in tqdm(titles_and_texts)
    ]
    training_data = [d for d in training_data if d.strip()]
    path_to_training_data = MALLET_OUTPUT_PATH + "/training.txt"
    path_to_formatted_training_data = MALLET_OUTPUT_PATH + "/mallet.training"

    lmw.import_data(
        MALLET_PATH,
        path_to_training_data,
        path_to_formatted_training_data,
        training_data,
    )

    for num_topics in NUM_TOPIC_VALUES:
        print("\n -- -- -- number of topics = " + str(num_topics) + " -- -- --\n")
        path_to_model = MALLET_OUTPUT_PATH + "/mallet.model." + str(num_topics)
        path_to_topic_keys = (
            MALLET_OUTPUT_PATH + "/mallet.topic_keys." + str(num_topics)
        )
        path_to_topic_distributions = (
            MALLET_OUTPUT_PATH + "/mallet.topic_distributions." + str(num_topics)
        )
        path_to_word_weights = (
            MALLET_OUTPUT_PATH + "/mallet.word_weights." + str(num_topics)
        )
        path_to_diagnostics = (
            MALLET_OUTPUT_PATH + "/mallet.diagnostics." + str(num_topics) + ".xml"
        )

        lmw.train_topic_model(
            MALLET_PATH,
            path_to_formatted_training_data,
            path_to_model,
            path_to_topic_keys,
            path_to_topic_distributions,
            path_to_word_weights,
            path_to_diagnostics,
            num_topics,
        )


def add_topic_distribution_to_combined_data_csv(df, num_topics):
    """
    For a specified value of K (num_topics), loads the doc-topic distributions, and saves that as added columns to the csv data.
    """
    topic_distributions = lmw.load_topic_distributions(
        MALLET_OUTPUT_PATH + "/mallet.topic_distributions." + str(num_topics)
    )
    topic_distributions = np.array(topic_distributions)
    for topic_ind in range(num_topics):
        df["Topic " + str(topic_ind)] = list(topic_distributions[:, topic_ind])
    df.to_csv(
        OUTPUT_DATA_WITH_TOPIC_DISTRIBUTION_BASEPATH
        + "all_texts_combined_with_topic_distribution_values_for_num_topics_"
        + str(num_topics)
        + ".csv"
    )


def save_url_to_topic_distributions_mapping(df, num_topics):
    """
    For a specified value of K (num_topics), loads the doc-topic distributions, and saves that as URL -> list of topic distribution values.
    """
    topic_distributions = lmw.load_topic_distributions(
        MALLET_OUTPUT_PATH + "/mallet.topic_distributions." + str(num_topics)
    )
    topic_distributions = np.array(topic_distributions)
    out = {}
    urls = list(df["url"])
    for i, url in tqdm(enumerate(urls)):
        out[url] = list(topic_distributions[i, :])
    pickle.dump(
        out,
        open(
            OUTPUT_DATA_WITH_TOPIC_DISTRIBUTION_BASEPATH
            + "url_to_topic_distribution_for_num_topics_"
            + str(num_topics)
            + ".pkl",
            "wb",
        ),
    )


if __name__ == "__main__":
    input_file_names = [
        x for x in os.listdir(INPUT_DEDUP_FILES_BASE_PATH) if "dedup" in x
    ]
    combined_df = create_and_save_combined_dataframe(input_file_names)
    topic_model_training(combined_df)
    for num_topics in NUM_TOPIC_VALUES:
        add_topic_distribution_to_combined_data_csv(combined_df, num_topics)
        save_url_to_topic_distributions_mapping(combined_df, num_topics)
