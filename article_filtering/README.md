# Filtering out irrelevant articles

We tried various filters to rule out articles not relevant for transgender issues, using a small annotated sample (see attempts in the jupyter notebooks). Ultimately, the best filter is a simple keyword-based one, using the presence or absence of the word 'gender' in the article. 

The performance of this keyword-based filter can be viewed in `03-keyword-filtering.ipynb`

Script that implements the filter is `final_pg_keyword_filter.py`: It uses the input data at `/home/pranavgoel/trans-fer-entropy/national_outlet_filtering_in_state_collections/data_with_national_outlets_removed_in_state_collections/` and stores the output (which is the final, filtered dataset to run our topic modeling on) at `/home/pranavgoel/trans-fer-entropy/article_filtering/transgender_filtered_data_files/`. 


---

## Earlier Attempts

### Zero Shot Filtering 

This pipeline is still under development, but is ready to be tested or used *cautiously*.

The primary matters left to make this fully functional are:
1. test it on some small, labeled subset of articles
2. test other prompt strings
3. try few shot, if we are able to label some articles

