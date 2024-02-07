"""
Here, we create and save a sample of 100 articles from across our dataset, for manual annotation: binary (1/0) labeling of each article as _relevant_ or not. An article is _relevant_ if it discusses transgender-specific-issues in any way, shape, or form. This is the code for creating the aforementioned annotation file, that'd be hand-coded by each annotator independently. 

Attributes:
    INPUT_DATA_PATH: Directory path to files with article data. 
    OUTPUT_FILEPATH: Path to save annotation file.  
    RANDOM_SEED: Seed for random sampling. 
    SAMPLE_SIZE: Number of articles to annotate/size of random sample. 
    
Functions:
    load_and_combine_article_data: Loads all the article files representing various media groups (the 6 states + nytimes+foxnews) and concatenates them as one dataframe, while retaining only the relevant columns. 
    : 
    
@author: Pranav Goel
"""

import os
import random

import numpy as np
import pandas as pd

INPUT_DATA_PATH = "/home/pranavgoel/trans-fer-entropy/national_outlet_filtering_in_state_collections/data_with_national_outlets_removed_in_state_collections/"

OUTPUT_FILEPATH = "/home/pranavgoel/trans-fer-entropy/internal_relevance_annotation/annotation_files/sample_for_relevance_annotation.csv"

RANDOM_SEED = 42

SAMPLE_SIZE = 100


def load_and_combine_article_data():
    """
    Loads all the article files representing various media groups (the 6 states + nytimes+foxnews) and concatenates them as one dataframe, while retaining only the relevant columns.

    OUTPUT:
        combined_df: combined data of all various state and national articles.
    """
    input_file_names = [x for x in os.listdir(INPUT_DATA_PATH) if ".csv" in x]
    assert len(input_file_names) == 7

    selected_cols = ["url", "title", "subtitle", "text"]

    all_dfs = []
    for f in input_file_names:
        df = pd.read_csv(INPUT_DATA_PATH + f)
        df = df[selected_cols]
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs)
    return combined_df


def create_and_save_annotation_sample(combined_df):
    """
    Creates and save random sample for annotation.

    INPUT:
        combined_df: combined data of all various state and national articles.
    """
    sample_df = combined_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    sample_df["Relevance_Label"] = [None for _ in range(SAMPLE_SIZE)]
    sample_df.to_csv(OUTPUT_FILEPATH, index=False)


if __name__ == "__main__":
    combined_df = load_and_combine_article_data()
    create_and_save_annotation_sample(combined_df)
