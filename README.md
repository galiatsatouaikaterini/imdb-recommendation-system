## Movie and TV Series Recommender Systems

- files ´preprocessing.py´ and ´dataset_exploration.ipynb´ for data preprocessing and cleaning
- file ´wikipedia_fetch.py´ used for adding plot's from wikipedia
- file contenbasedfiltering.py for three approaches to contentbased - filtering
- file collaborativefiltering.py for Neural Collaborative Filtering 


### General Information
- download the datasets from google drive and then do the preprocessing steps to get the dataset
- DATASETS:
    - tconst: list with movie/shows ids
    - imdb: all movies / shows info from imdb
    - imdb_small: preprocessed + cleaned data from imdb, used as a base to get plots from wikipedia
        - shape: (65672, 10)
    - imdb_with_plots: cleaned data from imdb_small with plots from wikipedia 
        - shape: (23442, 11)
    - user: all info about users
    - user/movie matrix: user id, movie id and rating of user 

- one file for content-based filtering and one for collaborative filtering
- use pytorch for ML stuff

### TO DO
- preprocessing / preparation:
    - create user rating dataset: Alif
    - finish contentbasedfiltering: Carina

- collaborative - filtering
    - finish the evaluation of the prediction: Kat
    - train the NCF on our dataset

- writing
    - write about the data
    - write about contentbasedfiltering
    - compare collaborative results of kaggle + our dataset
    - related work: deep NN
    - related work: conclusion (top n is content)
    - write results
    - write conclusion


- Updates
    - i did a bunch of preprocessing tasks that we wrote down last time
    - directors: they dont have their actual names but just an id so can we use this for content based filtering?
    - i found that dnn is a good technique for recommendation. should we proceed with this? i have some basic steps written in the overview file
    - i implemented some basic concepts of content based filtering but of course there is no "machine learning" in them (mainly to test things out and incorporate them in the end product)
    - the wikipedia api is taking forever because of the big dataset so i have an input question before starting the fetching process to ensure that if we have already run it once and have the imdb_with_plots dataset saved, then there is no point in running it again. We have to run the code in someone else's laptop cause my weak old laptop cannot handle it and it crashes all the time. After running it, we can upload the imdb_with_plots dataset on drive and then i can download it to continue without my laptop crushing (hopefully that will be my only problem)
    

