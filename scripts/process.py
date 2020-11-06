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

    entries = [
        {
            "title": title,
            "summary": summary,
            "low_dim_embedding": ldemb.tolist(),
            "link": link,
            "timestamp": timestamp
        } for title, summary, ldemb, link, timestamp in zip(titles, summaries, low_dim_embeddings, links, timestamps)
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
