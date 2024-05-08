from preprocessing import get_dataframe
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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


# ------------------ Binary Feature Matrix for genre column ----------------------

#  each movie is represented by a set of binary features that indicate whether the movie belongs to a certain genre or not
print("------------Binary Feature Matrix Approach ---------------")

genre_matrix = pd.get_dummies(imdb['genres'].str.split(",").apply(pd.Series).stack()).sum(level=0) # create the binary feature matrix for genre
print("Genre Matrix: ", genre_matrix.head()) # genre_matrix: each row corresponds to a movie and each column corresponds to a genre (values: 0, 1)

similarity = cosine_similarity(genre_matrix) # computing the cosine similarity matrix
print("Cosine similarity for binary feature matrix: ", similarity)

def get_recommendations(title, top_n): # function to get the recommended movies, n: number of movie recommendations
    
    idx = imdb[imdb['primaryTitle'] == title].index[0] # finding the index of the movie with the given title
    
    similarity_scores = list(enumerate(similarity[idx])) # getting the cosine similarity scores for the movie
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)  # sorting the similarity scores in descending order
    
    movie_indices = [i[0] for i in similarity_scores[1:top_n+1]] # getting the top_n movie indices
    
    return imdb['primaryTitle'].iloc[movie_indices] # return the top_n most similar movies
     

# test 
title = input("Enter the title of a movie: ")

while True:
    n = input("Enter the number of recommened movies that you want: ")
    try:
        user_number = int(n)
        print(f"You entered the integer: {user_number}")
        break
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

# get the recommended movies
print(f"Top {n} recommended movies using binary feature matrix: ")
print(get_recommendations(title))



# ---------------------------- Bag of Words for genre column ----------------------

# eventually we can use this method for the plots but for now I am testing it on the genre column again
print("------------Bag of Words Approach ---------------")

titles = imdb['primaryTitle'].tolist()
genres = imdb['genres'].str.split(",").tolist()

def create_bow(genre_list): # creating a bag of words representation for the title genres
    bow = {}
    for genre in genre_list:
        bow[genre] = 1
    return bow

bags_of_words = [create_bow(movie_genres) for movie_genres in genres] # creating a list of bags of words representations for the title genres

genre_df = pd.DataFrame(bags_of_words, index=titles).fillna(0) # creating a df to store the bags of words representation for the title genres
print("Genres df: ", genre_df)

cosine_similarity = cosine_similarity(genre_df) # calculating the cosine similarity matrix between the titles

similarity_df = pd.DataFrame(cosine_similarity, index=genre_df.index, columns=genre_df.index) # creating a df with the cosine similarity scores
print("Similarity df for bow: ", similarity_df)



# test 
title = input("Enter the title of a movie: ")

while True:
    n = input("Enter the number of recommened movies that you want: ")
    try:
        user_number = int(n)
        print(f"You entered the integer: {user_number}")
        break
    except ValueError:
        print("Invalid input. Please enter a valid integer.")


title_index = similarity_df.index.get_loc(title) # finding the index of the title in the similarity dataframe

# getting the top n most similar titles to the given title
top_n = similarity_df.iloc[title_index].sort_values(ascending=False)[1:n+1]

print(f'Top {n} similar movies to {title}:') # printing the top n most similar titles to the given title
print(top_n)


# ----------------------------- TF- IDF --------------------------------------

genres_combined = imdb['genres'].str.replace('|', ' ') # combining the genres for each title into a single string
print("Genres combined: ", genres_combined)

tfidf = TfidfVectorizer() # creating a TfidfVectorizer object to transform the title genres into a Tf-idf representation
tfidf_matrix = tfidf.fit_transform(genres_combined) 
print("TF-IDF matrix: ", tfidf_matrix)

cosine_similarity = cosine_similarity(tfidf_matrix) # calculating the cosine similarity matrix between the titles
     
similarity_df = pd.DataFrame(cosine_similarity, index=imdb['primaryTitle'], columns=imdb['primaryTitle']) # creating a df with the cosine similarity scores
print("Similarity df for tf- idf: ", similarity_df)


# test 
title = input("Enter the title of a movie: ")

while True:
    n = input("Enter the number of recommened movies that you want: ")
    try:
        user_number = int(n)
        print(f"You entered the integer: {user_number}")
        break
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

title_index = similarity_df.index.get_loc(title) # finding the index of the title in the similarity dataframe

# getting the top n most similar titles to the given titles
top_n = similarity_df.iloc[title_index].sort_values(ascending=False)[1:n+1]

# print the top n most similar titles to the given title
print(f'Top {n} similar movies to {title}:')
print(top_n)



# so now i have to combine these methods for the content based filtering model and also add other columns such as plots: but this model doesnt have "machine learning"


# BoW and TF-IDF are based on word frequencies and do not capture the semantic relationships between words

#  
