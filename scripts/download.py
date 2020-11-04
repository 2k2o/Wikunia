import feedparser
from glob import glob
import json
import os
import pathlib
import datetime
from tqdm import tqdm


def download_lang(lang):
    rss_feed_file = os.path.join("./configs/news_sources", lang+".json")
    out_root = "data/raw"

    now = datetime.datetime.now()

    out_dir = os.path.join(out_root, str(now.year)+"/" +
                        str(now.month)+"/"+str(now.day))

    out_path = os.path.join(out_dir, lang+".json")
    
    entries = []
    urls = []
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            entries = json.load(f)
            urls = [e["link"] for e in entries]
    else:
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(rss_feed_file, 'r') as f:
        feeds = json.load(f)

    for feed in tqdm(feeds):
        url = feed["feedUrl"].strip()
        newsfeed = feedparser.parse(url)
        for entry in newsfeed.entries:
            if "updated_parsed" in entry:
                key = "updated_parsed"
            elif "published_parsed" in entry:
                key = "published_parsed"
            else:
                continue

            if now.year == entry[key].tm_year and now.month == entry[key].tm_mon and now.day == entry[key].tm_mday:
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
