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


- Updates
    - i did a bunch of preprocessing tasks that we wrote down last time
    - directors: they dont have their actual names but just an id so can we use this for content based filtering?
    - i found that dnn is a good technique for recommendation. should we proceed with this? i have some basic steps written in the overview file
    - i implemented some basic concepts of content based filtering but of course there is no "machine learning" in them (mainly to test things out and incorporate them in the end product)
    - the wikipedia api is taking forever because of the big dataset so i have an input question before starting the fetching process to ensure that if we have already run it once and have the imdb_with_plots dataset saved, then there is no point in running it again. We have to run the code in someone else's laptop cause my weak old laptop cannot handle it and it crashes all the time. After running it, we can upload the imdb_with_plots dataset on drive and then i can download it to continue without my laptop crushing (hopefully that will be my only problem)
    

