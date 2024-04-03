import pickle
import networkx as nx
import pandas as pd
import urllib
# z = pickle.load(open('./string_alignment.pkl', 'rb'))
from urllib.parse import urlparse
import torch
prefixes = ['california', 'florida', 'illinois', 'ohio', 'newyork', 'nytimes_foxnews', 'texas',]
for prefix in prefixes:
    df = pd.read_csv('/data_users1/sagar/trans-fer-entropy/kw_filtered_data/' + prefix + '_article_texts_and_info_dedup.csv')
    

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, device=0, tokenizer=tokenizer)

for prefix in prefixes:
    df = pd.read_csv('/data_users1/sagar/trans-fer-entropy/kw_filtered_data/' + prefix + '_article_texts_and_info_dedup.csv')
    entities = []
    for idx, b in enumerate(df['text'].tolist()):
        if idx % 50 == 0:
            print(idx)
        res = nlp(b)
        entities.append(res)
    df['entities'] = entities 
    df.to_csv('/data_users1/asmithh/trans-fer-entropy/extra_analyses/kw_filtered_ner_{}_article_texts_and_info_dedup.csv'.format(prefix), sep='\t')
