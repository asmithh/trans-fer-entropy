"""
This script will filter articles to remove transgender-irrelavant articles by using a keyword based filter: where the keyword is 'gender'. 

Attributes: 
    KEYWORD_PATTERN: pattern for keyword for filtering articles. 
    INPUT_PATH: Directory path with deduplicated csv files, with state collections having national outlet articles removed.
    OUTPUT_PATH: Directory path to store the dataset files after filtering. 
    
Functions:
    keyword_filter: Get list of 1/0 labels for whether each article in a given dataframe has the KEYWORD_PATTERN in any component of the article text or not. 
    
@author: Pranav Goel (direct adaptation of code from Sagar Kumar)
"""

import os
import re

import pandas as pd


KEYWORD_PATTERN = r"gender"
INPUT_PATH = "/home/pranavgoel/trans-fer-entropy/national_outlet_filtering_in_state_collections/data_with_national_outlets_removed_in_state_collections/"
OUTPUT_PATH = "/home/pranavgoel/trans-fer-entropy/article_filtering/transgender_filtered_data_files/"


def keyword_filter(dataset):
    """
    Get list of 1/0 labels for whether each article in a given dataframe has the KEYWORD_PATTERN in any component of the article text or not.

    INPUT:
        dataset (pandas DataFrame()): input dataset file

    OUTPUT:
        keyword_labels (list): binary 1/0 labels of whether article text (any component of the text) includes the KEYWORD_PATTERN or not.
    """

    keyword_labels = []

    cols = dataset.columns

    if "text" in cols:
        column = dataset.text

    elif "subtitle" in cols:
        column = dataset.subtitle

    elif "title" in cols:
        column = dataset.title

    else:
        raise ValueError("No valid text columns.")

    for t in column:
        match = re.findall(KEYWORD_PATTERN, str(t), re.IGNORECASE)
        keyword_labels.append((len(match) > 0) * 1)

    return keyword_labels


if __name__ == "__main__":
    files = os.listdir(INPUT_PATH)
    for f in files:
        if f[-4:] == ".csv":
            print(f"Processing {f}")
            df = pd.read_csv(INPUT_PATH + f)
            labels = keyword_filter(df)
            df["gender_label"] = labels
            df_new = df[df.gender_label == 1]
            df_new.to_csv(OUTPUT_PATH + f, sep=",", index=False)
    print("New data files saved.")
