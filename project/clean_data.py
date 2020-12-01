import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
import itertools
from scipy.linalg import null_space
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from HSIC import hsic_gam
from numpy.linalg import svd


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
            if cast_count >= min_actors:
                movie_id = self.credits.loc[i, "movie_id"]
                self.movie_d[movie_id] = {
                    "title" : self.credits.loc[i, 'title'],
                    "revenue" : np.log(self.movies.loc[i, 'revenue'] + 1),
                    "cast" : list(cast_set)
                    }
        print(f"Found {len(self.movie_d)} movies.")

    def return_movies_df(self):
        df = pd.DataFrame.from_dict(self.movie_d, orient="index")
        mlb = MultiLabelBinarizer()
        cast_df = pd.DataFrame(mlb.fit_transform(df["cast"]),
                               columns=[f"a{k}" for k in mlb.classes_],
                               index=df.index)
        df = pd.merge(df, cast_df, left_index=True, right_index=True)
        del df['cast']
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


def covariance(Y, Z):
    # Y: n x dim(Y)
    # Z: n x dim(Z)
    # return: dim(Y) x dim(Z)
    return Y.T @ Z 

def correlation(Y, Z):
    Y = Y - Y.mean()
    Z = Z - Z.mean()
    vY = np.sqrt(np.sum(np.square(Y)))
    vZ = np.sqrt(np.sum(np.square(Z)))
    r = Y.T @ Z / (vY*vZ)
    return r.item()

# Tests if two random variables have 0 correlation
# using the fisherz transform
def fisherz_test(X1, X2):
    # X1: n x 1
    # X2: n x 1
    n = X1.shape[0]
    r = correlation(X1, X2)
    fisherz = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    cdf = norm.cdf(abs(fisherz), loc=0, scale=se)
    p = 2*(1-cdf)
    return p

def independence_test(E_YllZ, Z, alpha=0.01):
    p_list = []
    for i in range(Z.shape[1]):
        z = np.expand_dims(Z[:, i], 1)
        pval = hsic_gam(E_YllZ, z)
        #p = fisherz_test(E_YllZ, z)
        p_list.append(pval)

    # Fisher's Method
    c = len(p_list)
    p_list = [p+0.0000000001 for p in p_list]
    testStat = -2 * np.log(p_list).sum()
    pval = 1 - chi2.cdf(testStat, 2*c)
    print(pval)
    return pval < alpha


def generate_data(n=100):

    def noise(d=1, s=1, n=n):
        X = np.empty((n, d))
        for i in range(d):
            X[:, i] = np.random.uniform(low=-s, high=s, size=n)
        return X

    # Make Latent Vars
    L1 = noise()
    alpha = 4; beta=3; gamma=2; eta=1; theta=5; sigma=6
    L2 = L1 * alpha + noise()
    L3 = L1 * beta + L2 * sigma + noise()
    L4 = L1 * gamma + L2 * eta + L3 * theta + noise()

    # Make Observed Vars
    A = np.array([2] * 4)[:, None]
    B = np.array([3.4] * 4)[:, None]
    X1 = L1 @ A.T + L2 @ B.T + noise(4, 0.1)
    
    C = np.array([3] * 2)[:, None]
    X2 = L3 @ C.T + noise(2, 0.1)

    D = np.array([5] * 2)[:, None]
    X3 = L4 @ D.T + noise(2, 0.1)
    X = np.hstack([X1, X2, X3])
    X = pd.DataFrame(X)
    X.columns = [f"X{i+1}" for i in range(X.shape[1])]
    return X

def left_nullspace(X):
    U, S, V = svd(X.T, full_matrices=True, compute_uv=True)
    w = V.T[:,-1][:, None]
    return w


class GinModel():
    def __init__(self, data: pd.DataFrame):
        self.cols = data.columns
        self.X = data.to_numpy()

    def normalize_data(self):
        self.X = self.X - self.X.mean(axis=0)
        self.X = self.X / self.X.std(axis=0)

    def algo1(self, alpha=0.01):

        def _merge_sets(S):
            sets = S
            merged = True
            while merged:
                merged = False
                results = []
                while sets:
                    common, rest = sets[0], sets[1:]
                    sets = []
                    for x in rest:
                        if x.isdisjoint(common):
                            sets.append(x)
                        else:
                            merged = True
                            common |= x
                    results.append(common)
                sets = results
            return sets

        S = []
        n = 2
        P = self.X
        while True:
            print(f"Testing subsets of dim {n}...")
            subsets = itertools.combinations(range(P.shape[1]), n)
            for subset in subsets:
                subset = list(subset)
                Ymask = np.zeros(P.shape[1], dtype=bool)
                Ymask[subset] = True
                Y = P[:, Ymask]
                Z = P[:, ~Ymask]
                E_YZ = covariance(Y, Z)

                # Least Squares solution
                w = left_nullspace(E_YZ)
                #print(np.square(w.T @ E_YZ).sum())
                #exit(1)
                #w = null_space(E_YZ.T)

                # Check if w can be found in left null space
                #if w.size == 0:
                #    continue

                # Y: n x dim(Y)
                # w: dim(Y) x 1
                # E_YllZ: n x 1
                E_YllZ = Y @ w
                print(f"Testing {subset}...")
                indep = independence_test(E_YllZ, Z, alpha=alpha)
                if indep:
                    S.append(set(subset))
                    print(subset)

            # Merge overlapping sets
            if len(S) > 0:
                import IPython; IPython.embed(); exit(1)
                sets = _merge_sets(S)
            n += 1




# Read data
credits_f = "data/tmdb_5000_credits.csv"
movies_f = "data/tmdb_5000_movies.csv"
credits = pd.read_csv(credits_f)
movies = pd.read_csv(movies_f)

# Construct actors
actors = Actors(credits)
actors.construct_actors(max_actors=5)
actors.sort_actors()
actors.threshold_actors(num_actors=5)

# Construct movies
m = Movies(movies, credits, actors)
m.construct_movies(min_actors=2)
df = m.return_movies_df()
df.to_csv("data/cleaned.csv", index=False)

# Generate data ala GIN paper
X = generate_data(500)
X.to_csv("data/sim.csv", index=False)

# Simple lm
# model = run_linear_regression(df, "revenue", "title")

# Testing Independence Testing
#sigma = 0.2
#mean = [0, 0]
#cov = [[1, sigma], [sigma, 1]]
#p_list = []
#for i in range(1000):
#    X, Y = np.random.multivariate_normal(mean, cov, size=100).T
#    X = np.expand_dims(X, 1)
#    Y = np.expand_dims(Y, 1)
#    p = hsic_gam(X, Y)
#    p_list.append(p)
#plt.hist(p_list, bins=100)
#plt.savefig("plot.png")
#exit(1)


# Gin
#gin = GinModel(df.drop("title", axis=1))
#gin = GinModel(X)
#gin.normalize_data()
#gin.algo1(alpha=0.01)
