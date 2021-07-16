import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")

df = pd.read_csv("movie_metadata.csv")
print(df.shape)
print(df.columns)
df.isnull().sum()
df["color"].value_counts()
df["color"].fillna(df["color"].value_counts().index[1],inplace = True)
df["director_name"] = df["director_name"].replace(np.nan,"unknown")
df["actor_1_name"] = df["actor_1_name"].replace(np.nan,"unknown")
df["actor_2_name"] = df["actor_2_name"].replace(np.nan,"unknown")
df["actor_3_name"] = df["actor_3_name"].replace(np.nan,"unknown")
df["genres"] = df["genres"].apply(lambda x: str(x).replace("|", " "))
df["movie_title"] = df["movie_title"].str.lower()
df["director_name"] = df["director_name"].str.lower()
df["actor_1_name"] = df["actor_1_name"].str.lower()
df["actor_2_name"] = df["actor_2_name"].str.lower()
df["actor_3_name"] = df["actor_3_name"].str.lower()
df["genres"] = df["genres"].str.lower()
df["movie_title"] = df["movie_title"].apply(lambda x : x[:-1])
df["duration"].fillna(df["duration"].mean(),inplace = True)
df["director_facebook_likes"].fillna(df["director_facebook_likes"].mean(),inplace = True)
df["actor_1_facebook_likes"].fillna(df["actor_1_facebook_likes"].mean(),inplace = True)
df["actor_2_facebook_likes"].fillna(df["actor_2_facebook_likes"].mean(),inplace = True)
df["actor_3_facebook_likes"].fillna(df["actor_3_facebook_likes"].mean(),inplace = True)
df["gross"].fillna(df["gross"].mean(),inplace = True)
df["language"].fillna(df["language"].value_counts().index[0],inplace = True)
df["content_rating"].fillna("not_rated", inplace = True)

features = df[["director_name", "genres", "movie_title", "actor_1_name", "actor_2_name", "actor_3_name"]]
features["director_genre_actors"] = features["director_name"] + " " + features["genres"] + " " + features["actor_1_name"] + " " + features["actor_2_name"] + " " + features["actor_3_name"]
features = features.drop(["director_name", "actor_1_name", "actor_2_name", "actor_3_name", "genres"], axis = 1)
print(features.head())


from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer().fit_transform(features["director_genre_actors"])
print(vector.shape)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_Transformer = TfidfTransformer()
vector_matrix = tfidf_Transformer.fit_transform(vector)
print(vector_matrix.shape)

from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(metric = "cosine",n_neighbors = 20,radius = 1)
model.fit(vector_matrix)

from sklearn.metrics.pairwise import sigmoid_kernel
sig = sigmoid_kernel(vector_matrix,vector_matrix)

def recommendation(movie_name):
  if movie_name not in features["movie_title"].unique():
    return []
  else:
    i = features.loc[features["movie_title"] == movie_name].index[0] 
    sig_scores = list(enumerate(sig[i]))
    list_movie = sorted(sig_scores, key = lambda x: x[1], reverse = True)
    list_movie = list_movie[1:11]
    lst = []
    for i in range(len(list_movie)):
      j = list_movie[i][0]
      lst.append(features["movie_title"][j])
    return lst

print(recommendation("avatar"))

import pickle
pickle.dump(model, open("recomm.pkl", "wb"))