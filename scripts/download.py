import feedparser
from glob import glob
import json
import os
import datetime
from tqdm import tqdm

def date_of_entry(entry):
    if "updated_parsed" in entry:
        key = "updated_parsed"
    elif "published_parsed" in entry:
        key = "published_parsed"
    else:
        return datetime.datetime(1970, 1, 1, 0, 0, 0)

    e = entry[key]
    published_date = datetime.datetime(e.tm_year, e.tm_mon, e.tm_mday, e.tm_hour, e.tm_min, e.tm_sec)
    return published_date

def download_lang(lang):
    rss_feed_file = f"./configs/news_sources/{lang}.json"
    out_path = f"data/raw/{lang}.json"

    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(hours=24)

    # Load previous news entries
    entries = []
    urls = []
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            entries = json.load(f)
            urls = [e["link"] for e in entries]

    # Remove entries that are older than a day
    entries = [entry for entry in entries if date_of_entry(entry) > yesterday]

    # Get feeds
    with open(rss_feed_file, 'r') as f:
        feeds = json.load(f)

    # Add new entries younger than a day
    for feed in tqdm(feeds):
        url = feed["feedUrl"].strip()
        newsfeed = feedparser.parse(url)
        for entry in newsfeed.entries:
            if date_of_entry(entry) > yesterday:
                entry["feed"] = feed["name"]
                if entry.link not in urls:
                    entries.append(entry)
                    urls.append(entry.link)
                else:
                    # update if already present
                    i = urls.index(entry.link)
                    entries[i] = entry

    with open(out_path, 'w') as outfile:
        json.dump(entries, outfile)


if __name__ == "__main__":
    config_paths = glob("./configs/news_sources/*.json")
    # get language names
    langs = [os.path.splitext(os.path.basename(path))[0]
             for path in config_paths]

    for lang in langs:
        download_lang(lang)
