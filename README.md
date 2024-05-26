## Movie and TV Series Recommender Systems

- files preprocessing.py and dataset_exploration.ipynb for data preprocessing and cleaning
- file wikipedia_fetch.py used for adding plot's from wikipedia
- file contenbasedfiltering.py for three approaches to contentbased - filtering
- file collaborativefiltering.py for Neural Collaborative Filtering 
- file fakeuserbase.py for creating the user dataset
- file preprocess_user.ipynb for cleaning the user dataset

### Datasets
- tconst: list with movie/shows ids
- imdb: all movies / shows info from imdb
- imdb_small: preprocessed + cleaned data from imdb, used as a base to get plots from wikipedia
    - shape: (65672, 10)
- imdb_with_plots: cleaned data from imdb_small with plots from wikipedia 
    - shape: (23442, 11)
- user_movie_rating: user ratings created randomly and with bag of words algorithm to rate similar movies
    - shape: (271407, 3)
- user_movie_rating_cleaned: cleaned user_movie_rating, rounded some ratings + adjusted number of ratings for 1.0 and 10.0
    - shape: (271407, 3)
- movie_small: preprocessed kaggle dataset for movielens
    - shape: (913, 3)
- rating_small: preprocessed kaggle dataset for movielens
    - shape: (411184, 4)

### TO DO

- writing
    - write about the data: preprocessing kaggle dataset
    - compare collaborative results of kaggle + our dataset
    - related work: deep NN ? 
    - write results
    - write conclusion

