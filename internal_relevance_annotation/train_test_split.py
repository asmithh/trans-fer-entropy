"""
Splitting annotated sample data into training (for a few shot classifier) and testing (the few shot classifier); using either the consensus or majority label across three annotations on binary relevance of samples. Binary relevance is 1/0 labeling of each article as relevant or not. An article is relevant if it discusses transgender-specific-issues in any way, shape, or form. 

The few shot classifier will then be applied to our entire dataset (at /home/pranavgoel/trans-fer-entropy/national_outlet_filtering_in_state_collections/data_with_national_outlets_removed_in_state_collections/) to filter out irrelevant articles. 

Attributes:
    LABELED_ANNOTATIONS_PATH: Directory path to independently annotated sample files. 
    RANDOM_SEED: Seed for random sampling. 
    OUTPUT_TRAIN_FILEPATH: Path to save labeled training data file. 
    OUTPUT_TEST_FILEPATH: Path to save labeled test data file.  
    
Functions:
    obtain_aggregate_rating_counts: Returns aggregate rating counts (an array of shape number of samples (100) by number of categories (2)): each row is a sample, and a column is number of annotators with that particular rating or label (counts). 
    obtain_labels: Return list of labels -- one label (0/1) for each sample article. 
    
@author: Pranav Goel
"""

import os
import random

import numpy as np
import pandas as pd
import statsmodels.stats.inter_rater

LABELED_ANNOTATIONS_PATH = "/home/pranavgoel/trans-fer-entropy/internal_relevance_annotation/annotation_files/labeled/"

RANDOM_SEED = 33

OUTPUT_TRAIN_FILEPATH = "/home/pranavgoel/trans-fer-entropy/internal_relevance_annotation/train_sample_for_relevance_classifier.csv"

OUTPUT_TEST_FILEPATH = "/home/pranavgoel/trans-fer-entropy/internal_relevance_annotation/test_sample_for_relevance_classifier.csv"


def obtain_aggregate_rating_counts(annotation_df1, annotation_df2, annotation_df3):
    """
    Returns aggregate rating counts (an array of shape number of samples (100) by number of categories (2)): each row is a sample, and a column is number of annotators with that particular rating or label (counts).

    INPUT:
        annotation_df1: Annotated sample dataframe by annotator 1.
        annotation_df2: Annotated sample dataframe by annotator 2.
        annotation_df3: Annotated sample dataframe by annotator 3.

    OUTPUT:
        agg_rating_counts: Aggregate rating counts (number of annotators selecting 0/1 for each sample) -- 2d numpy array.
    """

    raters_data = np.array(
        [
            list(annotation_df1["Relevance_Label"]),
            list(annotation_df2["Relevance_Label"]),
            list(annotation_df3["Relevance_Label"]),
        ]
    ).T
    inter_rater_stats_table = statsmodels.stats.inter_rater.aggregate_raters(
        raters_data, n_cat=2
    )
    agg_rating_counts = inter_rater_stats_table[0]
    return agg_rating_counts


def obtain_labels(agg_rating_counts):
    """
    Return list of labels -- one label (0/1) for each sample article.

    INPUT:
        agg_rating_counts: Aggregate rating counts (number of annotators selecting 0/1 for each sample) -- 2d numpy array.

    OUTPUT:
        labels: list of 0/1 labels for the 100 sample articles
    """
    labels = []
    for i in range(agg_rating_counts.shape[0]):
        if 3 in agg_rating_counts[i]:
            if agg_rating_counts[i][0] == 3:
                labels.append(0)
            elif agg_rating_counts[i][1] == 3:
                labels.append(1)
        else:
            if agg_rating_counts[i][0] == 2:
                labels.append(0)
            elif agg_rating_counts[i][1] == 2:
                labels.append(1)
    assert len(labels) == 100
    return labels


if __name__ == "__main__":
    annotation_df1 = pd.read_csv(
        LABELED_ANNOTATIONS_PATH + "sample_for_relevance_annotation_YY.csv"
    )

    annotation_df2 = pd.read_csv(
        LABELED_ANNOTATIONS_PATH + "sample_for_relevance_annotation_labeled_SK.csv"
    )

    annotation_df3 = pd.read_csv(
        LABELED_ANNOTATIONS_PATH
        + "alyssa_annotations - sample_for_relevance_annotation.tsv",
        sep="\t",
    )
    annotation_df3.iloc[3, annotation_df3.columns.get_loc("Relevance_Label")] = 1.0
    annotation_df3 = annotation_df3.drop("Unnamed: 5", axis=1)
    annotation_df3["Relevance_Label"] = annotation_df3["Relevance_Label"].astype(int)

    assert (
        list(annotation_df1["url"])
        == list(annotation_df2["url"])
        == list(annotation_df3["url"])
    )

    agg_rating_counts = obtain_aggregate_rating_counts(
        annotation_df1, annotation_df2, annotation_df3
    )

    labels = obtain_labels(agg_rating_counts)
    labeled_df = annotation_df1[["url", "title", "subtitle", "text"]]
    labeled_df["label"] = labels

    train = labeled_df.sample(frac=0.5, random_state=RANDOM_SEED)
    test = labeled_df.drop(train.index)
    train.to_csv(OUTPUT_TRAIN_FILEPATH, index=None)
    test.to_csv(OUTPUT_TEST_FILEPATH, index=None)
