from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


df = pd.read_csv("all_movies.csv")

id = []
title = []
corpus = []
details_by_id = {}

for index, row in df.iterrows():
    id.append(str(row['id']))
    title.append(row['title'])
    corpus.append(row['corpus'])
    details_by_id[str(row['id'])] = {
        "title": row['title'], "corpus": row['corpus'], "backdrop": row['backdrop'], "poster": row['poster']}

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

genres = {}

for root, dirs, files in os.walk("./genres"):
    for file_name in files:
        genre = file_name[:-4]
        file_path = os.path.join(root, file_name)
        df = pd.read_csv(file_path)
        df = df.drop('corpus', axis=1)
        df = df.fillna('')
        data = df.to_dict('records')
        genres[genre] = data


def find_similar_corpus(query, tfidf_matrix):
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    sorted_indices = np.argsort(similarity_scores)[::-1]
    return sorted_indices, similarity_scores


def find_similar_movies(_ids, k):
    l = 4
    movies = []
    added_movie_ids = set(_ids)
    count = 0
    for _id in _ids:
        if count > 15:
            break
        _corpus = details_by_id[_id]["corpus"]
        indices, sim_scores = find_similar_corpus(_corpus, tfidf_matrix)
        for i in range(1, l):
            index = indices[i]
            movie_id = id[index]
            if movie_id and title[index] and movie_id not in added_movie_ids:
                count += 1
                added_movie_ids.add(movie_id)
                poster = details_by_id[movie_id]["poster"]
                backdrop = details_by_id[movie_id]["backdrop"]
                if not backdrop:
                    backdrop = ""
                if not poster:
                    poster = ""
                movies.append({
                    "id": movie_id,
                    "title": title[index],
                    "backdrop": backdrop,
                    "poster": poster
                })
            else:
                l += 1
    return movies


all_movies = []
df2 = pd.read_csv("all_movies.csv")
df2 = df2.drop('corpus', axis=1)
df2 = df2.fillna('')
data = df2.to_dict('records')
all_movies = data


@app.route("/get_recommendations/<int:k>/<ids>")
def similar_movies(ids, k):
    ids = ids.split("|")
    sim_movies = find_similar_movies(ids, k)
    return jsonify(sim_movies)


@app.route("/get_genre_movies/<genre>")
def genre_movies(genre):
    return jsonify(genres[genre])


@app.route("/get_all_movies")
def get_all_movies():
    return jsonify(all_movies)


if __name__ == "__main__":
    app.run(debug=True)

# flask --app app.py --debug run
