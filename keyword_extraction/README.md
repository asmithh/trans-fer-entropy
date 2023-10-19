Saved keyword files at: /home/pranavgoel/trans-fer-entropy/keyword_extraction/

Source article data files at: /home/pranavgoel/trans-fer-entropy/obtaining_news_collections/data/*_dedup.csv

Code used to extract and save dated keywords: extract_and_save_keywords_for_all_articles.ipynb

The keywords are contained in *_url_to_dated_keywords.pkl as dictionaries of the following format:

Key (of the dict) = URL; Value = [(timestamp, [(score, keyword/phrase), ...]), (), ...] 