import gzip
import pandas as pd
import json


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def parse_fields(line):
    data = json.loads(line)
    return{
        "book_id" : data['book_id'],
        "title" : data['title_without_series'],
        "ratings" : data['ratings_count'],
        "url" : data['url'],
        "cover_image" : data['image_url'],
        "num_pages" : data["num_pages"],
    }

book_titles = []
with gzip.open("goodreads_books.json.gz","r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        
        fields = parse_fields(line)

        try:
            ratings = int(fields['ratings'])
            num_pages = int(fields['num_pages'])
        except ValueError:
            continue
        if ratings > 15 :
            book_titles.append(fields)

titles = pd.DataFrame.from_dict(book_titles)
titles['ratings'] = pd.to_numeric(titles['ratings'])
titles['num_pages'] = pd.to_numeric(titles['num_pages'])

titles['mod_title'] = titles['title'].str.replace("[^a-zA-Z0-9 ]","",regex=True)

titles = titles[titles['mod_title'].str.len() >0]

# titles.to_json("./books_titles.json")

df = titles[titles['num_pages'] > 3]

tfidf = vectorizer.fit_transform(titles["mod_title"])

def make_clickable(val):
    return '<a target="_blank" href="{}">Goodreads</a>'.format(val, val)

def show_image(val):
    return '<a href="{}"><img src="{}" width=50></img></a>'.format(val, val)

# around 2 minutes 30 sec
def search_book(query,vectorizer):
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = titles.iloc[indices]
    results = results.sort_values("ratings", ascending=False)
    results.drop('mod_title',axis=1,inplace=True)
    return results

def pages_read_in_time(time_in_minutes, reading_speed_words_per_minute=225, words_per_page=275):
    total_words_read = time_in_minutes * reading_speed_words_per_minute
    pages_read = total_words_read / words_per_page
    return pages_read

def recommmend_books_by_time(time_in_minutes):
    pages = pages_read_in_time(time_in_minutes)
    return df[df['num_pages'] <=pages].sort_values(by=['num_pages','ratings'],ascending=False).head(20)
    