import pandas as pd
import numpy as np

movie = pd.read_csv('movie.csv')
rating = pd.read_csv('rating.csv')

# drop users with less than 500 movie ratings
rating_copy = rating.copy()
counts = rating_copy['userId'].value_counts()

val_drop = counts[(counts < 500)].index

df_filtered = rating_copy[~rating_copy['userId'].isin(val_drop)]

# drop users with over 5000 movie ratings
counts2 = df_filtered['userId'].value_counts()
val_drop2 = counts2[(counts2 > 5000)].index

df_filtered = df_filtered[~df_filtered['userId'].isin(val_drop2)]

# drop 1/4 of users randomly
np.random.seed(0)

# Get a list of unique users
unique_users = df_filtered['userId'].unique()

# Randomly select half of the users
users_to_keep = np.random.choice(unique_users, size=int(len(unique_users)/4), replace=False)

# Keep only the randomly selected users in the DataFrame
df_filtered = df_filtered[df_filtered['userId'].isin(users_to_keep)]

# drop movies with less than or 150 ratings
count_movies = df_filtered['movieId'].value_counts()

values_to_drop = count_movies[count_movies <= 150].index

df_filtered = df_filtered[~df_filtered['movieId'].isin(values_to_drop)]

# drop 1/3 of the movies randomly
np.random.seed(0)

# Get a list of unique users
unique_movies = df_filtered['movieId'].unique()

# Randomly select half of the users
movies_to_keep = np.random.choice(unique_movies, size=int(len(unique_movies)/3), replace=False)

# Keep only the randomly selected users in the DataFrame
df_filtered = df_filtered[df_filtered['movieId'].isin(movies_to_keep)]

# drop movies that aren't used anymore
df_movies_filtered = movie[movie['movieId'].isin(df_filtered['movieId'])]
print(df_movies_filtered.shape)

# print different users and movies
print(df_filtered['userId'].nunique(), " users in dataset")
print(df_filtered['movieId'].nunique(), " movies in dataset")

# save datasets
df_filtered.to_csv('rating_small.csv', index=False)
df_movies_filtered.to_csv('movie_small.csv', index=False)