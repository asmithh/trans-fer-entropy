- train_mallet_on_combined_article_text_data.py : This code borrows trains MALLET topic model(s) on combined article text data, saving all the outputs of the model, and also adding the topic probabilities to the combined data file. And saving the url to topic distribution mapping itself. Topics themselves will be interpreted and displayed (in terms of top words and top documents) in interpret_topic_modeling_output.ipynb.

- interpret_topic_modeling_output.ipynb: This notebook looks at the topics (in terms of top words and top documents) for each topic, to enable interpretation for each topic uncovered by various topic models trained and saved in train_mallet_on_combined_article_text_data.py
  
- requirements.txt: Python libraries needed to run the codes above. Python version used: 3.8.10