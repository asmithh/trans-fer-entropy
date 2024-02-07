Here, we create and save a sample of 100 articles from across our dataset, for manual annotation: binary (1/0) labeling of each article as _relevant_ or not. An article is _relevant_ if it discusses transgender-specific-issues in any way, shape, or form. 

`create_internal_relevance_annotation_sample.py`: Code for creating the aforementioned annotation file, that'd be hand-coded by each annotator independently.

`annotation_files/sample_for_relevance_annotation.csv`: File with article information, in order to label them for relevance. 

**Annotation instructions:** 

In each row, fill the cell in the last column titled `Relevance_Label` as either 1 or 0. 1 indicates the article talks about transgender people and/or transgender-specific issues. A mention of trans rights or transgender people in passing along with a bunch of other discussions in the article still makes it relevant for our purposes. Irrelevant articles won't mention or discuss trans issues at all, and the "trans" keyword would happen by-way of terminology used in fields like biotech, etc. 

After labeling all 100 articles with 1/0 in the last column, save your file (as a .csv -- in the same format) by adding an underscore followed by your initials at the end to the filename; for example, Pranav Goel will save the file as `sample_for_relevance_annotation_pg.csv` after labeling is complete. 

