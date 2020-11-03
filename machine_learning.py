from bs4 import BeautifulSoup
import json
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from sentence_transformers import util
from tqdm import tqdm

embedder = SentenceTransformerDocumentEmbeddings(
    'distiluse-base-multilingual-cased-v2')

file_path = "rss/2020/11/1/rss_de.json"

with open(file_path, 'r') as f:
    entries = json.load(f)

titles = [e['title'] for e in entries]
summaries = [e['summary'] for e in entries]
# tags = [[t["term"] for t in e['tags']] if "tags" in e else [] for e in entries]

data = [f"{t}. {s}" for t, s in zip(titles, summaries)]
# Clean data by filtering HTML tags
data = [BeautifulSoup(d, features="lxml").get_text() for d in data]

entries = [Sentence(d, use_tokenizer=True) for d in tqdm(data)]

# calculate embeddings
for s in tqdm(entries):
    embedder.embed(s)

embeddings = [s.embedding.detach().numpy() for s in entries]

reducer = TSNE(n_components=2)

X_trans = reducer.fit_transform(embeddings)


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
