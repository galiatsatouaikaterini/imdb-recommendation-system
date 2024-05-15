import pandas as pd
import numpy as np

imdb_small = pd.read_csv("imdb_small.csv")
print(imdb_small.shape)
#print(imdb_small.head())

#imdb_no_shorts = imdb_small[~imdb_small['titleType'].isin(values_to_exclude_titleType))]

titles = pd.read_csv("title.basics.tsv", sep = '\t', low_memory=False) 
titles.dropna()

print(titles['titleType'].unique())