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

imdb.to_csv('imdb.csv', index=False) 

# for the generation of the user data, we want the tconst column as a csv: import it to mockaroo
titles['tconst'].to_csv('tconst.csv', index=False) 


def get_dataframe(): # function to get dfs to another python file
    print("Getting the df...")
    return imdb

# --------------------- wikipedia api ----------------------------

def get_plots(imdb):
    # get all the titles of the movies that we want to extract the plots
    wiki_titles = imdb['primaryTitle'] # use the primary title column

    # create a list of all the names that the section might be called
    possibles = ['Plot','Synopsis','Plot synopsis','Plot summary', 
                'Story','Plotline','The Beginning','Summary',
                'Content','Premise']

    # sometimes those possible names have 'Edit' latched onto the end due to user error on wikipedia
    # in that case, it will be 'PlotEdit' so it's easier to make another list that acccounts for that
    possibles_edit = [i + 'Edit' for i in possibles]

    #then merge those two lists together
    all_possibles = possibles + possibles_edit

    print("Starting the fetching plot process. This might take a while due to the size of the dataset, be patient...")

    title_plots = [] # initialize list to store plots

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

        title_plots.append({'primaryTitle': i, 'plot': plot_})

    # create a df with the fetched plots
    plots_df = pd.DataFrame(title_plots)

    # Merge with the original DataFrame, using 'primaryTitle' as the key
    merged_df = imdb.merge(plots_df, on='primaryTitle', how='left')
    merged_df = merged_df.dropna(subset=['plot']) # dropping rows with null plots

    return merged_df


# ask if the user wants to run the plot fetching process
proceed = input("Do you want to proceed with fetching movie plots? (yes/no): ").strip().lower()

if proceed == 'yes':
    imdb_with_plots = get_plots(imdb)
    print("Fetching completed.")
    print("IMDB with Plots: ", imdb_with_plots)
    imdb_with_plots.to_csv('imdb_with_plots.csv', index=False) 
else:
    print("Process cancelled.")
