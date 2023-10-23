import pandas as pd

df_pink_slime = pd.read_csv('pink_slime_domains.csv')

fnames = [
    'california_article_texts_and_info_dedup.csv',
    'florida_article_texts_and_info_dedup.csv',
    'newyork_article_texts_and_info_dedup.csv',
    'ohio_article_texts_and_info_dedup.csv',
    'illinois_article_texts_and_info_dedup.csv',
    'texas_article_texts_and_info_dedup.csv',
    'nytimes_foxnews_article_texts_and_info_dedup.csv',
]

def get_domain(url):
    """
    Extracts top-level domain from a given URL
    """
    url_after_http = url.split('://')[1]
    domain = url_after_http.split('/')[0]
    if 'www.' in domain:
        domain.replace('www.', '')
    return domain

for fname in fnames:
    df_news = pd.read_csv('/home/pranavgoel/trans-fer-entropy/obtaining_news_collections/data/{}'.format(fname))
    len_orig = len(df_news)

    df_news['domain'] = df_news['url'].apply(get_domain)
    df_merged = pd.merge(df_news, df_pink_slime, left_on='domain', right_on='domain', how='right')
    df_merged = df_merged.dropna(subset=['publish_date', 'network'])
