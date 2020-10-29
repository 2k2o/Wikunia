from bs4 import BeautifulSoup
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
# from sklearn.manifold import TSNE
import plotly.graph_objects as go
# import spacy
# import numpy as np
import umap

file_path = "rss/2020/10/29/rss_de.json"

with open(file_path, 'r') as f:
    entries = json.load(f)

titles = [e['title'] for e in entries]
# summaries = [e['summary'] for e in entries]
tags = [[t["term"] for t in e['tags']] if "tags" in e else [] for e in entries]

entries = [f"{t} {' '.join(tags)}" for t, tags in zip(titles, tags)]
# entries = [f"{t} {s} {' '.join(tags)}" for t, s, tags in zip(titles, summaries, tags)]
entries = [BeautifulSoup(e, "html.parser").get_text() for e in entries]


# nlp = spacy.load("de_core_news_sm")
# X = np.array([np.mean([t.vector for t in nlp(e)], axis=0) for e in entries])

# vectorizer = CountVectorizer(binary=True)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(entries)
# print(vectorizer.vocabulary_)

# dim_red = TruncatedSVD(n_components=2)
dim_red = umap.UMAP()

X_trans = dim_red.fit_transform(X)

def color(title):
    if "corona" in title.lower():
        return "#ff0000"
    else:
        return "#000000"

fig = go.Figure(data=go.Scatter(x=X_trans[:, 0],
                                y=X_trans[:, 1],
                                mode='markers',
                                marker_color=[color(t) for t in titles],
                                text=titles))

fig.update_layout(title='News')
fig.show()
