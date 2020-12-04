import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


class Actors():

    def __init__(self, credits):
        self.credits = credits
        self.actors = {}

    # Find popular actors
    def construct_actors(self, max_actors=10):
        for i in range(self.credits.shape[0]):
            cast_list = json.loads(self.credits.loc[i, "cast"])
            n = len(cast_list)
            for j in range(min(n, max_actors)):
                v = cast_list[j]
                cast_id = v["cast_id"]
                name = v["name"]
                if cast_id not in self.actors:
                    self.actors[cast_id] = {"name": name, "count": 1}
                else:
                    self.actors[cast_id]["count"] += 1
        print(f"Found {len(self.actors)} unique actors.")

    # Sort actors by count
    def sort_actors(self):
        self.actors = dict(sorted(self.actors.items(), 
                key=lambda item: -item[1]["count"]))

    # Threshold them
    def threshold_actors(self, num_actors=10):
        if num_actors < 0:
            return 

        new_actors = {}
        for i, item in enumerate(self.actors.items()):
            if i >= num_actors:
                break
            k, v = item
            new_actors[k] = v
        self.actors = new_actors
        print(f"Selected top {num_actors} actors.")

class Movies():
    def __init__(self, movies, credits, actors):
        self.movies = movies 
        self.credits = credits
        self.actors = actors
        self.actor_set = set(actors.actors.keys())
        self.movie_d = {}

    def construct_movies(self, min_actors=2):
        for i in range(self.movies.shape[0]):
            cast_list = json.loads(self.credits.loc[i, "cast"])
            cast_set = set([d["cast_id"] for d in cast_list])
            cast_set = cast_set.intersection(self.actor_set)
            cast_count = len(cast_set)

            genre_list = json.loads(self.movies.loc[i, "genres"])
            genre_list = [d["name"] for d in genre_list]
            if cast_count >= min_actors:
                movie_id = self.credits.loc[i, "movie_id"]
                self.movie_d[movie_id] = {
                    "title" : self.credits.loc[i, 'title'],
                    "revenue" : np.log(self.movies.loc[i, 'revenue'] + 1),
                    "cast" : list(cast_set),
                    "genre": genre_list
                    }
        print(f"Found {len(self.movie_d)} movies.")

    def return_movies_df(self):
        df = pd.DataFrame.from_dict(self.movie_d, orient="index")
        mlb = MultiLabelBinarizer()
        cast_df = pd.DataFrame(mlb.fit_transform(df["cast"]),
                               columns=[f"a{k}" for k in mlb.classes_],
                               index=df.index)
        print(f"{cast_df.shape[1]} actor columns.")
        genre_df = pd.DataFrame(mlb.fit_transform(df["genre"]),
                                columns=[f"g{k}" for k in mlb.classes_],
                                index=df.index)
        print(f"{genre_df.shape[1]} genre columns.")
        df = pd.merge(df, cast_df, left_index=True, right_index=True)
        df = pd.merge(df, genre_df, left_index=True, right_index=True)
        del df['cast']
        del df['genre']
        return df

def run_linear_regression(df, y_col, remove_x):
    if isinstance(remove_x, str):
        remove_x = [remove_x]
    y = df[y_col]
    remove_cols = [y_col]
    remove_cols.extend(remove_x)
    X = df.drop(remove_cols, axis=1)
    reg = LinearRegression().fit(X, y)
    print(reg.score(X, y))
    return reg


if __name__ == "__main__":

    # Read data
    credits_f = "data/tmdb_5000_credits.csv"
    movies_f = "data/tmdb_5000_movies.csv"
    credits = pd.read_csv(credits_f)
    movies = pd.read_csv(movies_f)
    #import IPython;IPython.embed();exit(1)
    
    # Construct actors
    actors = Actors(credits)
    actors.construct_actors(max_actors=5)
    actors.sort_actors()
    actors.threshold_actors(num_actors=-1)
    
    # Construct movies
    m = Movies(movies, credits, actors)
    m.construct_movies(min_actors=2)
    df = m.return_movies_df()
    print(df.columns)
    df.to_csv("data/cleaned.csv", index=False)

