import pandas as pd

# reading the .tsv file
titles = pd.read_csv("title.basics.tsv", sep = '\t', low_memory=False) 
titles.dropna()
print(titles.head())

ratings = pd.read_csv("title.ratings.tsv", sep = '\t', low_memory=False)
ratings.dropna()
print(ratings.head())

crew = pd.read_csv("title.crew.tsv", sep = '\t', low_memory=False)
crew.dropna()
print(crew.head())

# remove adult movies from the titles dataset
titles = titles[titles['isAdult'] != 1] # isAdult (boolean) - 0: non-adult title; 1: adult title

# drop column ‘isAdult’
titles.drop('isAdult', axis=1,  inplace=True)
print(titles.head())

#genres_unique = titles['genres'].unique() # genres (string array) – includes up to three genres associated with the title
#print(genres_unique)
titles = titles[(titles['genres'] != 'Biography') & (titles['genres'] != 'Talk-Show')] #trying to limit the titles to movies, series

# drop column 'endYear'
titles.drop('endYear', axis=1,  inplace=True) # endYear (YYYY) – TV Series end year. ‘\N’ for all other title types

# drop column 'writers' from crew dataset
crew.drop('writers', axis=1, inplace=True)

# left- join the title and ratings dataset
titles_ratings = pd.merge(titles, ratings, on='tconst', how='left')
print(titles_ratings.head())

# for the generation of the user data, we want the tconst column as a csv: import it to mockaroo
titles['tconst'].to_csv('tconst.csv', index=False) 



# wikipedia api

import requests 
import wikipedia
import numpy as np

#language_code = 'en'
#search_query = 'solar system'
#number_of_results = 1
#headers = {
#  'Authorization': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI5NzcwYWE1ODA4Y2ExYzU0NmEyNDkxNTFkOTkyODBhNSIsImp0aSI6IjAwMmIzNGQwYjhhNTRhMDczNjVkYjQyMGI1OWQyMzAxNTA2MzI2ZWRlOWMwOTYyOTdkNTM3NzIxYTQ5ZmY1Yjk5OTgyZjJhYWNkN2JhZjhlIiwiaWF0IjoxNzE0NjQwMTI2LjkxNzkxNywibmJmIjoxNzE0NjQwMTI2LjkxNzkyMiwiZXhwIjozMzI3MTU0ODkyNi45MTYwNzMsInN1YiI6Ijc1NTYxMzcwIiwiaXNzIjoiaHR0cHM6Ly9tZXRhLndpa2ltZWRpYS5vcmciLCJyYXRlbGltaXQiOnsicmVxdWVzdHNfcGVyX3VuaXQiOjUwMDAsInVuaXQiOiJIT1VSIn0sInNjb3BlcyI6WyJiYXNpYyIsImNyZWF0ZWVkaXRtb3ZlcGFnZSIsImVkaXRwcm90ZWN0ZWQiXX0.bt5x9st3Jifsf205MNCuYPlUoZ2AAvZa8O24-ha36FRkcYP0WaruWTOXkF8k8dltAgH-15VLFJE5H9Iq1Izqw43I_aKU1pIjRDazXlQRMu5ifmVdUTXvrHSfIPijgMAgI43nZLdM92TbhdnRAe5L5n_GNTG5geVnnI35qP4LRiAjxuZ_H9d5JHb4ZRclRGyRduIF4DUrsxHCej_8SINLjxH23ttbHsGQgoa-flmx381uTHut7NJ3FvnKaBAM6i3at4alxYvJhNAg-Be0p17qhTe_ko7KSkfWpCek3paS22swJH4rxkMWZQI0liBbLZ6BQxEon15uxnSBYxqHm5aCzsSPbJsKgu8zHecLqNe9Cq6ouAH7xLeIvLj_3i1ItpFbFcniMOsXTBqauqq1AY0wShhUlmbpeXPIICmlU_FwhdOwTi1ut_A54wmZ_CjOGrxf3RT8_tvf4psYSb9bBvPbkzCe2FgXfLLGrD0h3UpopiEe0ww3Mj8Sb45G8d6_SYrYFFOhCJB3azZlJuwA4erYy4xHkpvW2HLYPOOgD-J3XESVAKzZqMDsE3-i5gyWE0xRf0s74x8g86BmJiH6osLYLZi4QzutfLrK0Di1IOTAIwKo0U8tNhX_UfrdJOiaqVwkPVHGBeamSytuEt1zp9KtOPUsd1ghiEihgjBgAqWdouM',
 # 'User-Agent': 'Y9770aa5808ca1c546a249151d99280a5'
#}

#base_url = 'https://api.wikimedia.org/core/v1/wikipedia/'
#endpoint = '/search/page'
#url = base_url + language_code + endpoint
#parameters = {'q': search_query, 'limit': number_of_results}
#response = requests.get(url, headers=headers, params=parameters)

# get all the titles of the movies that we want to extract the plots
wiki_titles = ['Barbie']

# create a list of all the names that the section might be called
possibles = ['Plot','Synopsis','Plot synopsis','Plot summary', 
             'Story','Plotline','The Beginning','Summary',
            'Content','Premise']

# sometimes those possible names have 'Edit' latched onto the end due to user error on wikipedia
# in that case, it will be 'PlotEdit' so it's easiest just to make another list that acccounts for that
possibles_edit = [i + 'Edit' for i in possibles]

#then merge those two lists together
all_possibles = possibles + possibles_edit

# fetch plots
for i in wiki_titles:
# loading the page once and save it as a variable, otherwise it will request the page every time
# always do a try, except when pulling from the API, in case it gets confused by the title
    try:
        wik = wikipedia.WikipediaPage(i[0])
    except:
        wik = np.NaN

# a new try, except for the plot
    try:
        # for all possible titles in all_possibles list
        for j in all_possibles:
            if wik.section(j) != None: # if that section does exist, i.e. it doesn't return 'None'
                plot_ = wik.section(j).replace('\n','').replace("\'","")  #then that's what the plot is! Otherwise try the next one!
    except: # if none of those work, or if the page didn't load from above, then plot equals np.NaN
        plot= np.NaN

