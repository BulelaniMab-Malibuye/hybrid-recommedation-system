import pandas as pd
import numpy as np
import pickle
from surprise import Reader, Dataset
from surprise import SVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv', sep=',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1, inplace=True)

# Load the pre-trained SVD model
model = pickle.load(open('resources/models/SVD.pkl', 'rb'))

# Load the movie content data
content_df = pd.read_csv('resources/data/movie_content.csv')

def predict_ratings(item_id):
    """Predict ratings given a movie ID.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df, reader)
    trainset = load_df.build_full_trainset()

    predictions = []
    for user_id in trainset.all_users():
        predictions.append(model.predict(uid=user_id, iid=item_id, verbose=False))

    return predictions

def get_similar_users(movie_list):
    """Find users with similar high ratings for the given list of movies.

    Parameters
    ----------
    movie_list : list
        Three favorite movies selected by the app user.

    Returns
    -------
    list
        User IDs of users with similar high ratings for each movie.

    """
    user_ids = []
    for movie_id in movie_list:
        predictions = predict_ratings(item_id=movie_id)
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_user_ids = [pred.uid for pred in predictions[:10]]
        user_ids.extend(top_user_ids)

    return user_ids

def get_content_based_recommendations(movie_list):
    """Get content-based recommendations based on movie metadata.

    Parameters
    ----------
    movie_list : list
        Three favorite movies selected by the app user.

    Returns
    -------
    list
        Titles of the top-n content-based movie recommendations.

    """
    vectorizer = TfidfVectorizer(stop_words='english')
    content_matrix = vectorizer.fit_transform(content_df['description'].values.astype('U'))
    indices = pd.Series(content_df.index, index=content_df['title']).drop_duplicates()

    movie_indices = [indices[movie] for movie in movie_list]
    cosine_sim = cosine_similarity(content_matrix[movie_indices], content_matrix)

    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:51]
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = content_df['title'].iloc[movie_indices].tolist()

    return recommended_movies

def collab_model(movie_list, top_n=10):
    """Perform hybrid collaborative and content-based filtering.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : int
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations for the user.

    """
    collaborative_recommendations = collab_model(movie_list, top_n=top_n)
    content_based_recommendations = get_content_based_recommendations(movie_list)

    recommended_movies = collaborative_recommendations + content_based_recommendations
    recommended_movies = list(set(recommended_movies))[:top_n]

    return recommended_movies
