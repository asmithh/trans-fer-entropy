Here, we create and save a sample of 100 articles from across our dataset, for manual annotation: binary (1/0) labeling of each article as _relevant_ or not. An article is _relevant_ if it discusses transgender-specific-issues in any way, shape, or form. We then split annotated samples into train-test set, for training a few-shot classifier. This few shot classifier will then be applied to our entire dataset (at`/home/pranavgoel/trans-fer-entropy/national_outlet_filtering_in_state_collections/data_with_national_outlets_removed_in_state_collections/`) to filter out irrelevant articles. 

`create_internal_relevance_annotation_sample.py`: Code for creating the aforementioned annotation file, that'd be hand-coded by each annotator independently.

`annotation_files/sample_for_relevance_annotation.csv`: File with article information, for annotation (instructions below). 

`analyze_annotator_agreement.ipynb`: Notebook analyzing the inter-annotator-agreement between the three annotators on binary relevance. 

`train_test_split.py`: Splitting annotated sample data into training (for few-shot classifier) and testing (the few-shot classifier). 

`train_sample_for_relevance_classifier.csv`: Labeled training data file (for few-shot classifier). 

`test_sample_for_relevance_classifier.csv`: Labeled test data file (for few-shot classifier). 

**Annotation instructions:** 

In each row, fill the cell in the last column titled `Relevance_Label` as either 1 or 0. 1 indicates the article talks about transgender people and/or transgender-specific issues. A mention of trans rights or transgender people in passing along with a bunch of other discussions in the article still makes it relevant for our purposes. Irrelevant articles won't mention or discuss trans issues at all, and the "trans" keyword would happen by way of terminology used in fields like biotech, etc. 

After labeling all 100 articles with 1/0 in the last column, save your file (as a .csv -- in the same format) by adding an underscore followed by your initials at the end to the filename; for example, Pranav Goel will save the file as `sample_for_relevance_annotation_pg.csv` after labeling is complete. 

