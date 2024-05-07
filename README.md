## Movie and TV Series Recommender Systems

- add summary / stuff to do here


### General Information
- download the datasets from google drive and then do the preprocessing steps to get the dataset
- DATASETS:
    - tconst: list with movie/shows ids
    - imdb: all movies / shows info from imdb and plots from wikipedia
    - user: all info about users
    - user/movie matrix: user id, movie id and rating of user 

- one file for content-based filtering and one for collaborative filtering
- use pytorch for ML stuff

### TO DO
- preprocessing / preparation:
    - create one type called movie for every type of movie and one for the tv series: DONE
    - explore the values of TitleType and Genres to drop unwanted ones: DONE
    - get wikipedia plots : DONE
    - join all movie datasets into one after preprocessing (add crew) : DONE
    - create user dataset with random info about users
    - create user rating dataset

- collaborative - filtering
    - Colaborative Filtering 
    - User based Filtering
    - Item Based Collaborative Filtering
    - Singular ValuE Decomposition

    - test Item Based Collaborative Filtering with test and training split

- content - filtering
    - tf-idf and cosine similarity for plots
        - spacy or other NLP to use or sklearn library
    - think about evaluation of this method
        - test movies and expected recommendations like sequels

