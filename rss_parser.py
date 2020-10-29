import feedparser
import json
import os
import pathlib
import datetime
from tqdm import tqdm
now = datetime.datetime.now()

file_name = "rss_de.json"
folder = "rss/"+str(now.year)+"/"+str(now.month)+"/"+str(now.day)
path = folder + "/" + file_name

l = []
urls = []
if os.path.exists(path):
    with open(path, "r") as f:
        l = json.load(f)
        urls = [e["link"] for e in l]

with open(file_name, 'r') as f:
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
                l.append(entry)
            else:
                # update if already present
                i = urls.index(entry.link)
                l[i] = entry


pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
with open(path, 'w') as outfile:
    json.dump(l, outfile)
