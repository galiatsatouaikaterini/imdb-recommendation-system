from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# import dataset with plots
imdb = pd.read_csv("imdb_with_plots.csv")

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

#applying the pre-processing to the 'plot' column
imdb_copy = imdb.copy()
imdb_copy['plot'] = imdb_copy['plot'].apply(preprocess_plot)

#create a bag of words model
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(imdb_copy['plot'])

# Function to find the cosine similarity for a given movie title
def find_similar_movies(movie_title, num_similar, bow_matrix, imdb_copy):
    
    # Get the index of the movie
    movie_idx = imdb_copy[imdb_copy['primaryTitle'] == movie_title].index[0]
    
    # Get the BoW vector for the movie
    movie_vector = bow_matrix[movie_idx]
    
    # Compute cosine similarity between the movie vector and all other movie vectors
    similarity_matrix = cosine_similarity(movie_vector, bow_matrix)
    
    # Get similarity scores and corresponding movie titles
    similarity_scores = list(enumerate(similarity_matrix[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    similar_movies = []
    for idx, score in similarity_scores[1:num_similar + 1]:  # Skip the first one as it will be the movie itself
        similar_movie_title = imdb_copy.iloc[idx]['primaryTitle']
        similar_movies.append((imdb_copy.iloc[idx]['tconst'], similar_movie_title, score))
    
    return similar_movies

# Create random users and ratings
num_users = 20000
users = [f"user_{i}" for i in range(num_users)]
movies = imdb_copy['tconst'].values
ratings_data = []

for user in users:
    # Select a random movie
    random_movie = random.choice(movies)
    random_movie_title = imdb_copy[imdb_copy['tconst'] == random_movie]['primaryTitle'].values[0]
    initial_rating = round(random.uniform(1, 10), 1)
    
    # Add the initial rating to the dataset
    ratings_data.append((user, random_movie, initial_rating))
    
    # Find a random number of similar movies between 2 and 20
    num_similar_movies = random.randint(2, 20)
    similar_movies = find_similar_movies(random_movie_title, num_similar_movies, bow_matrix, imdb_copy)
    
    # Rate similar movies with small increment/decrement
    for movie_id, movie_title, _ in similar_movies:
        rating_adjustment = round(random.uniform(-1, 1), 1)
        similar_rating = min(max(initial_rating + rating_adjustment, 1), 10)  # Ensure rating is between 1 and 10
        ratings_data.append((user, movie_id, similar_rating))
    
    # Give a random small number of similar movies very different ratings
    num_very_different_ratings = random.randint(0, 3)  # Random small number
    for _ in range(num_very_different_ratings):
        if similar_movies:
            movie_id, movie_title, _ = random.choice(similar_movies)
            very_different_rating = round(random.uniform(1, 10), 1)
            ratings_data.append((user, movie_id, very_different_rating))
    

# Step 4: Create the final dataset
ratings_df = pd.DataFrame(ratings_data, columns=['userID', 'movieID', 'rating'])

# Save to a CSV file
ratings_df.to_csv('user_movie_ratings.csv', index=False)
print("Dataset created successfully!")