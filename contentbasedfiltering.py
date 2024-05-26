import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# get dataset with movie plots
imdb = pd.read_csv("imdb_with_plots.csv")

# Steps:

# import data from preprocessing
# feature extraction: Extract meaningful features from movies. This can include genres, descriptions (using NLP techniques), cast, director, user ratings, etc.
# vectorize the text to numeric: Convert these features into numerical vectors. For text, you could use embeddings from models like Word2Vec, GloVe, or even transformers (e.g., BERT). For categorical data like genres, use one-hot encoding or embeddings.
# recommendation of top n movies


# ------------------ Binary Feature Matrix for startYear, runtimeMinutes, genres, averageRating column ----------------------

def get_binary_matrix(df, col):
    if col == 'genres':
        # transform genre into list
        df['genres'] = df['genres'].str.split(",")
        df_filtered = df[['tconst', col]]
    
        # divide lists into separate rows to get dummies properly divided (Action = True for [Action] and [Action, Thriller])
        df_exploded = df_filtered.explode(col)
        
        # get_dummies to convert the column to binary values (True, False)
        binary_matrix = pd.get_dummies(df_exploded[col])
        
        # join matrices together
        df_merged = pd.concat([df_exploded, binary_matrix], axis=1)
        
        # group by 'tconst' and sum the binary columns to get one row per movie again
        binary_matrix = df_merged.groupby('tconst').sum().reset_index()

        # drop unneccessary columns to have similar size to other matrices
        binary_matrix.drop('tconst', axis=1, inplace=True)
        binary_matrix.drop(col, axis=1, inplace=True)

    else:
        # get_dummies to convert the column to binary values (True, False)
        binary_matrix = pd.get_dummies(df[col])

    return binary_matrix

def get_recommendations_binary(imdb, binary_matrix, title, user_number):
    sim_scores = []
    idx = imdb[imdb['primaryTitle'] == title].index[0]
    similarity = cosine_similarity(binary_matrix)
    similarity_scores = list(enumerate(similarity[idx])) # getting the cosine similarity scores for the movie
    sim_scores.append(similarity_scores)

    index_scores = {}

    for inner_list in sim_scores: # get dictionary of all scores for each movie
        for index, score in inner_list:
            if index in index_scores:
                index_scores[index].append(score)
            else:
                index_scores[index] = [score]

    mean_scores = []
    for index, scores in index_scores.items():
        mean_score = sum(scores) / len(scores)
        mean_scores.append([index, mean_score])

    mean_scores = sorted(mean_scores, key=lambda x: x[1], reverse=True)  # sorting the similarity scores in descending order
    for idx, score in mean_scores[1:user_number+1]:  # Skip the first one as it will be the movie itself
        similar_movie_title = imdb.iloc[idx]['primaryTitle']
        print(f"Movie: {similar_movie_title}, Similarity Score: {score:.4f}")

# ---------------------------- Bag of Words for plot column ----------------------

# import nltk and stemmer for pre-processing
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

# removing stop words and performing stemming/lemmatization
def preprocess_plot(text):
    tokens = text.split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

# applying the pre-processing to the 'plot' column
imdb_copy = imdb.copy()
#imdb_copy['plot'] = imdb_copy['plot'].apply(preprocess_plot) 

def get_bow_matrix(df):
    # create a bag of words model
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(df['plot'])
    return bow_matrix


# ----------------------------- TF - IDF for plot columnn --------------------------------------

def get_tf_idf(df):
    # create a tf idf matrix
    tfidf = TfidfVectorizer(stop_words='english') # creating a TfidfVectorizer object to transform the title genres into a Tf-idf representation and removing stop words
    tfidf_matrix = tfidf.fit_transform(df['plot'])
    return tfidf_matrix


# ----------------------------- recommendation function --------------------------------------

def find_similar_movies(movie_title, matrix, df, n):
   
    # get the index of the movie
    movie_idx = df[df['primaryTitle'] == movie_title].index[0]
    
    # get the vector for the movie
    movie_vector = matrix[movie_idx]
    
    # compute cosine similarity between the movie vector and all other movie vectors
    similarity_matrix = cosine_similarity(movie_vector, matrix)
    
    # get similarity scores and corresponding movie titles
    similarity_scores = list(enumerate(similarity_matrix[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # print top n similar movies
    for idx, score in similarity_scores[1:n+1]:  # Skip the first one as it will be the movie itself
        similar_movie_title = df.iloc[idx]['primaryTitle']
        print(f"Movie: {similar_movie_title}, Similarity Score: {score:.4f}")


# ----------------------------- user input --------------------------------------

title = input("Enter the title of a movie: ")
# check if the movie title exists in the DataFrame
while True:
    title = input("Enter the title of a movie: ")
    if title in imdb['primaryTitle'].values:
        break
    else:
        print(f"Movie title {title} not found. Please enter a valid title.")

while True:
    n = input("Enter the number of recommened movies that you want: ")
    try:
        user_number = int(n)
        print(f"You entered the integer: {user_number}")
        break
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

print(f"Top {n} movies similar to '{title}':")
print()
print("------------ Binary Feature Matrix Approach ---------------")
# choose features to compare
columns = [ "startYear", "runtimeMinutes", "genres", "averageRating"]
for col in columns:
    binary_matrix = get_binary_matrix(imdb, col)
get_recommendations_binary(imdb, binary_matrix, title, user_number)

print()
print("------------ Bag of Words Approach ---------------")
bow_matrix = get_bow_matrix(imdb)
find_similar_movies(title, bow_matrix, imdb, user_number)

print()
print("------------ TF-IDF Approach ---------------")
tf_idf_matrix = get_tf_idf(imdb)
find_similar_movies(title, tf_idf_matrix, imdb, user_number)
