import pandas as pd
import re
import wikipedia
import numpy as np

# --------------------- importing data --------------------------

# reading the .tsv file
titles = pd.read_csv("title.basics.tsv", sep = '\t', low_memory=False) 
titles.dropna()
print("titiles dataset:", titles.head())

ratings = pd.read_csv("title.ratings.tsv", sep = '\t', low_memory=False)
ratings.dropna()
print("ratings dataset:", ratings.head())

crew = pd.read_csv("title.crew.tsv", sep = '\t', low_memory=False)
crew.dropna()
print("crew dataset:", crew.head())


# ---------------------- preprocessing --------------------------

# titles df: remove adult movies from the titles dataset
titles = titles[titles['isAdult'] != 1] # isAdult (boolean) - 0: non-adult title; 1: adult title

# drop column ‘isAdult’
titles.drop('isAdult', axis=1,  inplace=True)

# trying to limit the titles to genres related to movies and series
unique_genres = titles['genres'].unique()
print("Unique genres:", unique_genres)

# drop genres: videoGame, video, tvPilot, tvSpecial
values_to_exclude_genres = ['Biography', 'Talk-Show'] # values to exclude from the 'titleType' column

regex_pattern = '|'.join(values_to_exclude_genres) # constructing a regex pattern to match any of the specified genres

titles = titles[~titles['genres'].str.contains(regex_pattern, regex=True, na=False)]

# drop column 'endYear'
titles.drop('endYear', axis=1,  inplace=True) # endYear (YYYY) – TV Series end year. ‘\N’ for all other title types


# create one type for movies & one for tv series: now there are multiple types for each category and that makes the dataset complicated 
unique_title_types = titles['titleType'].unique() # first we need to check all the possible types for movies
print("Unique title types:", unique_title_types) # titleType (string) – the type/format of the title (e.g. movie, short, tvseries, tvepisode, video, etc)

titles.loc[titles['titleType'].isin(['movie', 'short', 'tvShort', 'tvMovie']), 'titleType'] = 'movie' # convert all movie types  to 'movie'
titles.loc[titles['titleType'].isin(['tvSeries', 'tvEpisode', 'tvMiniSeries']), 'titleType'] = 'tvseries' # convert all tv series types  to 'tvseries'


# drop titleType: videoGame, video, tvPilot, tvSpecial
values_to_exclude_titleType = ['videoGame', 'video', 'tvPilot', 'tvSpecial'] # values to exclude from the 'titleType' column
# the df excluding these values
titles = titles[~titles['titleType'].isin(values_to_exclude_titleType)]


# crew df: drop column 'writers' 
crew.drop('writers', axis=1, inplace=True)


# --------------------- JOIN datasets --------------------------------

# left- join the title and ratings dataset
titles_ratings = pd.merge(titles, ratings, on='tconst', how='left')
imdb = imdb_full = pd.merge(titles_ratings, crew, on='tconst', how='left')
print("IMDB dataset: ", imdb.head())
print('imdb shape: ', imdb.shape)

imdb.to_csv('imdb.csv', index=False) 

# --------------------- preproces joined IMDB dataset --------------------------------

# drop movies/series with less than 1000 ratings
imdb_small = imdb.copy().loc[imdb['numVotes']>= 1000]

# drop duplicates
imdb_small.drop_duplicates(subset=['primaryTitle', 'startYear'], keep='last', inplace=True)

# drop short movies/series in genre
imdb_small = imdb_small[~imdb_small['genres'].apply(lambda x: 'Short' in x)]

# drop movies/series after 2020 or before 1960
imdb_small = imdb_small.loc[imdb_small['startYear'] <= '2020']
imdb_small = imdb_small.loc[imdb_small['startYear'] >= '1960']

print("IMDB dataset: ", imdb.head())
print('imdb_small shape: ',imdb_small.shape)

imdb_small.to_csv('imdb_small.csv', index=False)

# for the generation of the user data, we want the tconst column as a csv: import it to mockaroo
imdb_small['tconst'].to_csv('tconst.csv', index=False) 
