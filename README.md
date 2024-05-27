## Movie and TV Series Recommender Systems

### Files
- `preprocessing.py` for data preprocessing and cleaning
- `wikipedia_fetch.py` used for adding plot's from wikipedia
- `contenbasedfiltering.py` for three approaches to contentbased - filtering
- `collaborativefiltering.py` for Neural Collaborative Filtering 
- `fakeuserbase.py` for creating the user dataset
- `user_preprocess.py` for cleaning user dataset
- `movielens_downsizing.py` for downsizing the kaggle movielens dataset

### Datasets
- title.basics.tsv: IMDB non-commercial dataset used for preprocessing
- title.crew.tsv: IMDB non-commercial dataset used for preprocessing
- title.ratings.tsv: IMDB non-commercial dataset used for preprocessing

- movie.csv: movielens dataset from kaggle
- rating.csv: movielens dataset from kaggle

- imdb.csv: all movies / shows info from imdb afte preprocessing
    - shape: (8890417, 10)
- imdb_small.csv: preprocessed + cleaned data from imdb, used as a base to get plots from wikipedia
    - shape: (65672, 10)
- imdb_with_plots.csv: cleaned data from imdb_small with plots from wikipedia 
    - shape: (23442, 11)
- user_movie_ratings.csv: user ratings created randomly and with bag of words algorithm to rate similar movies
    - shape: (271407, 3)
- user_movie_rating_cleaned.csv: cleaned user_movie_ratings, rounded some ratings + adjusted number of ratings for 1.0 and 10.0
    - shape: (271407, 3)
- movie_small.csv: downsized kaggle dataset for movielens
    - shape: (913, 3)
- rating_small.csv: downsized kaggle dataset for movielens
    - shape: (411184, 4)


### Setup

#### content-based filtering
1. use preprocessing.py to create imdb.csv dataset and imdb_small.csv
2. use wikipedia_fetch.py to create the imdb_with_plots.csv dataset
3. run contentbasedfiltering.py to get recommendations

#### collaborative filtering
1. use preprocessing.py to create imdb.csv dataset and imdb_small.csv
2. use wikipedia_fetch.py to create the imdb_with_plots.csv dataset
3. use fakeuserbase.py to create user dataset
4. use user_preprocess.py to clean user dataset
5. use movielens_downsizing.py to reduce movielens
5. run collaborativefiltering.py to train and evaluate NCF model
