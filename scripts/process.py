from glob import glob
import os
from bs4 import BeautifulSoup
import json
from sklearn.manifold import TSNE
from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from tqdm import tqdm
from datetime import datetime
import pathlib
from scipy.optimize import linear_sum_assignment
import numpy as np

print("Loading model")
embedder = SentenceTransformerDocumentEmbeddings(
    'distiluse-base-multilingual-cased-v2')


def get_timestamps(entries):
    timestamps = []
    for entry in entries:
        if "updated_parsed" in entry:
            key = "updated_parsed"
        elif "published_parsed" in entry:
            key = "published_parsed"
        else:
            timestamps.append(datetime.now())
            continue
        dt = datetime(*entry[key][:6])
        timestamps.append(datetime.timestamp(dt))
    return timestamps

def process_lang(lang):
    file_path = f"data/raw/{lang}.json"
    out_path = f"data/processed/{lang}.json"

    with open(file_path, 'r') as f:
        entries = json.load(f)

    print("Extracting data")
    titles = [e['title'] for e in entries]
    # Clean data by filtering HTML tags
    summaries = [BeautifulSoup(
        e['summary'], features="lxml").get_text() for e in entries]
    links = [e['link'] for e in entries]
    timestamps = get_timestamps(entries)
    feeds = [e['feed'] for e in entries]

    data = [f"{t}. {s}" for t, s in zip(titles, summaries)]

    entries = [Sentence(d, use_tokenizer=True) for d in tqdm(data)]
    

    print("Calculating embeddings")
    for s in tqdm(entries):
        embedder.embed(s)

    embeddings = [s.embedding.detach().numpy() for s in entries]

    print("Reducing dimensions")
    reducer = TSNE(n_components=2)
    low_dim_embeddings = reducer.fit_transform(embeddings)

    print("Fitting data to grid")
    max_x, max_y = np.max(low_dim_embeddings, axis=0)
    min_x, min_y = np.min(low_dim_embeddings, axis=0)
    x_spread = max_x - min_x
    y_spread = max_y - min_y

    # bring the data into a wider representation
    if y_spread > x_spread:
        low_dim_embeddings[:, [0, 1]] = low_dim_embeddings[:, [1, 0]]
        
        min_x, min_y = min_y, min_x
        max_x, max_y = max_y, max_x

        x_spread, y_spread = y_spread, x_spread

    # build a hexagon grid
    grid_size = 200
    grid_x = np.arange(grid_size)*1.5
    grid_y = np.arange(grid_size)*np.sin(np.deg2rad(60))*0.5
    grid = np.array([
        (x if i % 2 == 0 else x+0.75, y)
        for x in grid_x for i, y in enumerate(grid_y)
    ])

    low_dim_embeddings = (low_dim_embeddings - np.min(low_dim_embeddings, axis=0))
    low_dim_embeddings *= np.min(np.max(grid, axis=0)/np.max(low_dim_embeddings, axis=0))

    cost_matrix = np.array([[np.sqrt(np.sum((d - g)**2))
                             for g in grid] for d in low_dim_embeddings])
    _, col_indices = linear_sum_assignment(cost_matrix)
    grid_embeddings = grid[col_indices, :]


    entries = [
        {
            "title": title,
            "summary": summary,
            "embedding": emb.tolist(),
            "link": link,
            "timestamp": timestamp,
            "feed": feed
        } for title, summary, emb, link, timestamp, feed in zip(titles, summaries, grid_embeddings, links, timestamps, feeds)
    ]

    print("Saving processed data")
    with open(out_path, 'w') as outfile:
        json.dump(entries, outfile)


if __name__ == "__main__":
    config_paths = glob("./configs/news_sources/*.json")
    # get language names
    langs = [os.path.splitext(os.path.basename(path))[0]
             for path in config_paths]

    for lang in langs:
        print(f"Processing language: {lang}")
        process_lang(lang)
