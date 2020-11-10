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


def save_entries(lang, entries):
    now = datetime.now()
    out_dir = f"data/processed/{now.year}/{now.month}/{now.day}"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, lang+".json")
    with open(out_path, 'w') as outfile:
        json.dump(entries, outfile)

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
    now = datetime.now()
    file_path = f"data/raw/{now.year}/{now.month}/{now.day}/{lang}.json"

    with open(file_path, 'r') as f:
        entries = json.load(f)

    print("Extracting data")
    titles = [e['title'] for e in entries]
    # Clean data by filtering HTML tags
    summaries = [BeautifulSoup(
        e['summary'], features="lxml").get_text() for e in entries]
    # tags = [[t["term"] for t in e['tags']] if "tags" in e else [] for e in entries]
    links = [e['link'] for e in entries]
    timestamps = get_timestamps(entries)

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
    grid_size = 100
    grid = np.array([(x, y) for x in range(grid_size) for y in range(grid_size)])
    max_x, max_y = np.max(low_dim_embeddings, axis=0)
    min_x, min_y = np.min(low_dim_embeddings, axis=0)

    x_spread = max_x - min_x
    y_spread = max_y - min_y

    max_spread = np.max([x_spread, y_spread])
    low_dim_embeddings[:, 0] = (low_dim_embeddings[:, 0] - min_x) * \
        (grid_size-1) / max_spread
    low_dim_embeddings[:, 1] = (low_dim_embeddings[:, 1] - min_y) * \
        (grid_size-1) / max_spread

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
            "timestamp": timestamp
        } for title, summary, emb, link, timestamp in zip(titles, summaries, grid_embeddings, links, timestamps)
    ]

    print("Saving processed data")
    save_entries(lang, entries)


if __name__ == "__main__":
    config_paths = glob("./configs/news_sources/*.json")
    # get language names
    langs = [os.path.splitext(os.path.basename(path))[0]
             for path in config_paths]

    for lang in langs:
        print(f"Processing language: {lang}")
        process_lang(lang)
