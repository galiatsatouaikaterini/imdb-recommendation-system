# from preprocessing import get_dataframe
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# import df from preprocessing
#imdb = get_dataframe()
#print(imdb.head())

imdb = pd.read_csv("imdb_with_plots.csv")

# Steps:

# import data from preprocessing

# feature extraction: Extract meaningful features from movies. This can include genres, descriptions (using NLP techniques), cast, director, user ratings, etc.

# vectorize the text to numeric: Convert these features into numerical vectors. For text, you could use embeddings from models like Word2Vec, GloVe, or even transformers (e.g., BERT). For categorical data like genres, use one-hot encoding or embeddings.

# recommendation of top n movies


# for starters I am testing everything for one column only: genre


# ------------------ Binary Feature Matrix for each column ----------------------
def get_binary_matrix(df, col):
    
    if col == 'genres':
        # transform genre into list
        df['genres'] = df['genres'].str.split(",")
        df_filtered = df[['tconst',col]]
    
        # Explode the column into separate rows to get dummies properly divided (Action = True for [Action] and [Action, Thriller])
        df_exploded = df_filtered.explode(col)
        
        # Use get_dummies to convert the column to binary values (True, False)
        binary_matrix = pd.get_dummies(df_exploded[col])
        
        # Join the binary matrix back to the exploded DataFrame
        df_exploded = pd.concat([df_exploded, binary_matrix], axis=1)
        
        # Group by 'tconst' and sum the binary columns to get one row per movie again
        binary_matrix = df_exploded.groupby('tconst').sum().reset_index()

        # drop unneccessary columns to have similar size to other matrices
        binary_matrix.drop('tconst', axis=1, inplace=True)
        binary_matrix.drop(col, axis=1, inplace=True)

    else:
    
        # Use get_dummies to convert the column to binary
        binary_matrix = pd.get_dummies(df[col])

    return cosine_similarity(binary_matrix)


def get_recommendations(title, top_n): # function to get the recommended movies, n: number of movie recommendations
    
    idx = imdb[imdb['primaryTitle'] == title].index[0] # finding the index of the movie with the given title
    
    similarity = get_binary_matrix(imdb, 'averageRating')
    similarity_scores = list(enumerate(similarity[idx])) # getting the cosine similarity scores for the movie
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)  # sorting the similarity scores in descending order
    movie_indices = [i[0] for i in similarity_scores[1:int(top_n)+1]] # getting the top_n movie indices
    
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
print(get_recommendations(title, n))



# ---------------------------- Bag of Words for genre column ----------------------

# eventually we can use this method for the plots but for now I am testing it on the genre column again
print("------------Bag of Words Approach ---------------")

#import nltk and stemmer for pre-processing
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
stemmer = PorterStemmer()

#removing stop words and performing stemming/lemmatization
def preprocess_plot(text):
    tokens = text.split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

"""
titles = imdb['primaryTitle'].tolist()
genres = imdb['genres'].str.split(",").tolist()
"""
#applying the pre-processing to the 'plot' column
imdb_copy = imdb.copy()
#imdb_copy['plot'] = imdb_copy['plot'].apply(preprocess_plot) 

#create a bag of words model
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(imdb_copy['plot'])

# convert the BoW matrix to a dataframe
"""
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
bow_df['title'] = imdb_copy['primaryTitle']
"""
"""
def create_bow(genre_list): # creating a bag of words representation for the title genres
    bow = {}
    for genre in genre_list:
        bow[genre] = 1
    return bow

bags_of_words = [create_bow(movie_genres) for movie_genres in genres] # creating a list of bags of words representations for the title genres

genre_df = pd.DataFrame(bags_of_words, index=titles).fillna(0) # creating a df to store the bags of words representation for the title genres
print("Genres df: ", genre_df)
"""
# Function to find the cosine similarity for a given movie title
def find_similar_movies(movie_title, bow_matrix, imdb_copy):
    # Check if the movie title exists in the DataFrame
    if movie_title not in imdb_copy['primaryTitle'].values:
        print(f"Movie title '{movie_title}' not found.")
        return None
    
    # Get the index of the movie
    movie_idx = imdb_copy[imdb_copy['primaryTitle'] == movie_title].index[0]
    
    # Get the BoW vector for the movie
    movie_vector = bow_matrix[movie_idx]
    
    # Compute cosine similarity between the movie vector and all other movie vectors
    similarity_matrix = cosine_similarity(movie_vector, bow_matrix)
    
    # Get similarity scores and corresponding movie titles
    similarity_scores = list(enumerate(similarity_matrix[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Print top 10 similar movies
    print(f"Top 10 movies similar to '{movie_title}':")
    for idx, score in similarity_scores[1:11]:  # Skip the first one as it will be the movie itself
        similar_movie_title = imdb_copy.iloc[idx]['primaryTitle']
        print(f"Movie: {similar_movie_title}, Similarity Score: {score}")

"""
# cosine_similarity = cosine_similarity(bow_df) # calculating the cosine similarity matrix between the titles

# similarity_df = pd.DataFrame(cosine_similarity, index=genre_df.index, columns=genre_df.index) # creating a df with the cosine similarity scores
# print("Similarity df for bow: ", similarity_df)

"""

# test 
title = input("Enter the title of a movie: ")
"""
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
"""

find_similar_movies(title, bow_matrix, imdb_copy)

# ----------------------------- TF- IDF --------------------------------------

genres_combined = imdb['genres'].str.replace('|', ' ') # combining the genres for each title into a single string
print("Genres combined: ", genres_combined)

tfidf = TfidfVectorizer(stop_words='english') # creating a TfidfVectorizer object to transform the title genres into a Tf-idf representation and removing stop words
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
# maybe we can compare the above methods with the machine learning model to have some kind of comparison but apart from that we need to read papers about evaluation methods
# BoW and TF-IDF are based on word frequencies and do not capture the semantic relationships between words

#  
