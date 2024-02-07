## Filtering out articles from national outlets, present in our six state media collections

### Labeled Domain Dataset

The domain classification file at `/home/pranavgoel/trans-fer-entropy/national_outlet_filtering_in_state_collections/domain_classification_data/combined_clean_local_national_domain_classifications.csv` contains domains with three classes of labels: `local`, `national`, and `INCONSISTENT`. The labelings are pooled from six EXISTING domain classification datasets (references below -- please cite those papers in any project using this domain classification file). If labels assigned to the same domains are different (local in one case, national in another) across two or more of the existing domain classification datasets, the label `INCONSISTENT` is used. We will use this combined domain classification file to filter out national outlets from state media collections. 

The six existing datasets used to create the aforementioned classification file: 

1. A list of over 5000 US news domains and their social media accounts (https://github.com/ercexpo/us-news-domains). Citation: `Clemm von Hohenberg, B., Menchen-Trevino, E., Casas, A., Wojcieszak, M. (2021). A list of over 5000 US news domains and their social media accounts. https://doi.org/10.5281/zenodo.7651047`
2. Local News Social Media Dataset (https://github.com/sTechLab/local-news-dataset). Citation: `Understanding Local News Social Coverage and Engagement at Scale during the COVID-19 Pandemic, Marianne Aubin Le Quéré, Ting-Wei Chiang, Mor Naaman, Sixteenth International AAAI Conference on Web and Social Media, 2022.`
3. Google news auditing dataset (https://osf.io/hwuxf/?view_only=3fa7499661df487689031e11b8ea20b4). Citation: `Fischer, Sean, Kokil Jaidka, and Yphtach Lelkes. 2022. “National News Outlets Are Favored over Local News Outlets in News Aggregator Results.” OSF. December 22. osf.io/hwuxf.`
4. Local News Dataset 2018 (https://github.com/yinleon/LocalNewsDataset). Citation: `Leon Yin. (2018). yinleon/LocalNewsDataset: Initial release (V1.0). Zenodo. https://doi.org/10.5281/zenodo.1345145`
5. NELA-Local (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GFE66K). Citation: `Horne, Benjamin D., et al. "NELA-Local: A Dataset of US Local News Articles for the Study of County-Level News Ecosystems." Proceedings of the International AAAI Conference on Web and Social Media. Vol. 16. 2022.`
6. ABYZ dataset (http://www.abyznewslinks.com/unite.htm). Citation: mention `ABYZ News Links` and include the link. 

Note: We acknowledge Kaicheng Yang, who downloaded, compiled, and cleaned all the six aforementioned datasets and combined them. 

### Code

- filter_national_outlets_from_state_collections.py: This script will display the numbers before and after removal of articles by national outlets in each of the six state collections, and save filtered files for later usage in the project. 