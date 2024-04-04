- `train_mallet_on_combined_article_text_data.py`: This code borrows trains MALLET topic model(s) on combined article text data, saving all the outputs of the model, and also adding the topic probabilities to the combined data file. And saving the url to topic distribution mapping itself. Topics themselves will be interpreted and displayed (in terms of top words and top documents) in `interpret_topic_modeling_output.ipynb`.

- `train_mallet_on_combined_article_text_data_apr_jun_2023_data.py`: Above but for our OG dataset of Apr-June 2023.

- `train_mallet_on_combined_article_text_data_apr_jun_2023_data_after_removing_trans_irrelevant_articles.py`: Above code but for filtered dataset (with both national outlets removed from state media, and irrelevant articles removed using keyword filter).

- `train_mallet_on_combined_article_text_data_apr_jun_2023_data_after_removing_national_outlets_in_state_media.py`: Above but on intermediate dataset with national outlets removed from state media, mostly used just for some internal checks and purposes. 

- `interpret_topic_modeling_output.ipynb`: This notebook looks at the topics (in terms of top words and top documents) for each topic, to enable interpretation for each topic uncovered by various topic models trained and saved in `train_mallet_on_combined_article_text_data.py`

- `interpret_topic_modeling_output_prior_data_apr_jun_2023.ipynb`: This notebook looks at the topics (in terms of top words and top documents) for each topic, to enable interpretation for each topic uncovered by various topic models trained and saved in `train_mallet_on_combined_article_text_data_apr_jun_2023_data.py`

- `interpret_topic_modeling_apr_jun_2023_post_irrelevant_article_filtering.ipynb`: This notebook looks at the topics (in terms of top words and top documents) for each topic, to enable interpretation for each topic uncovered by various topic models trained and saved in `train_mallet_on_combined_article_text_data_apr_jun_2023_data_after_removing_trans_irrelevant_articles.py` -- it also proposes topic labels for the research team to view and discuss. 

- `interpret_topic_modeling_output_prior_data_apr_jun_2023_after_national_outlet_filtering.ipynb`: This notebook looks at the topics (in terms of top words and top documents) for each topic, to enable interpretation for each topic uncovered by various topic models trained and saved in `train_mallet_on_combined_article_text_data_apr_jun_2023_data_after_removing_national_outlets_in_state_media.py`
  
- `requirements.txt`: Python libraries needed to run the codes above. Python version used: 3.8.10