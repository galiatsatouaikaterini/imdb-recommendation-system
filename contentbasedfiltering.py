from preprocessing import get_dataframe
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# import df from preprocessing
imdb = get_dataframe()
print(imdb.head())

# Steps:

# import data from preprocessing

# feature extraction

# vectorize the text to numeric

# similarity score

# recommendation of top n movies


# for starters I am testing everything for one column only: genre


# ------------------ binary feature matrix for genre column ----------------------

#  each movie is represented by a set of binary features that indicate whether the movie belongs to a certain genre or not

genre_matrix = pd.get_dummies(imdb['genres'].str.split(",").apply(pd.Series).stack()).sum(level=0) # create the binary feature matrix for genre
print("Genre Matrix: ", genre_matrix.head()) # genre_matrix: each row corresponds to a movie and each column corresponds to a genre (values: 0, 1)

similarity = cosine_similarity(genre_matrix) # computing the cosine similarity matrix
print("Cosine similarity: ", similarity)

def get_recommendations(title, top_n): # function to get the recommended movies, n: number of movie recommendations
    
    idx = imdb[imdb['primaryTitle'] == title].index[0] # finding the index of the movie with the given title
    
    similarity_scores = list(enumerate(similarity[idx])) # getting the cosine similarity scores for the movie
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)  # sorting the similarity scores in descending order
    
    movie_indices = [i[0] for i in similarity_scores[1:top_n+1]] # getting the top_n movie indices
    
    return imdb['primaryTitle'].iloc[movie_indices] # return the top_n most similar movies
     

# test the recommendation system
title = input("Enter the title of a movie: ")

while True:
    top_n = input("Enter the number of recommened movies that you want: ")
    try:
        user_number = int(top_n)
        print(f"You entered the integer: {user_number}")
        break
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

# get the recommended movies
print(f"Top {top_n} recommended movies:")
print(get_recommendations(title))



# ---------------------------- bag of words for genre column ----------------------

# eventually we can use this method for the plots but for now I am testing it on the genre column again

titles = movies['title'].tolist()
genres = movies['genres'].str.split("|").tolist()
