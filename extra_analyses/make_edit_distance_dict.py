import pandas as pd
import pickle
from fast_edit_distance import edit_distance

df = pd.read_csv('/home/pranavgoel/trans-fer-entropy/topic_modeling/prior_data_apr_jun_2023/all_texts_combined.csv')


d = {}
for idx1, row1 in df.iterrows():
    print(idx1 / len(df))
    id1 = (row1['media_name'], row1['url'])
    txt1 = row1['text']
    d[id1] = {}
    for idx2, row2 in df.iterrows():
        id2 = (row2['media_name'], row2['url'])
        txt2 = row2['text']
        res = edit_distance(txt1, txt2, max_ed=700)
        d[id1][id2] = res
pickle.dump(d, open('string_alignment_1500.pkl', 'wb'))
