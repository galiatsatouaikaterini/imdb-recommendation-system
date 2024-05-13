import wikipedia
import pandas as pd
import numpy as np

def get_plots(imdb):
    # get all the titles of the movies that we want to extract the plots
    wiki_titles = imdb['primaryTitle'] # use the primary title column

    # create a list of all the names that the section might be called
    possibles = ['Plot','Synopsis','Plot synopsis','Plot summary', 
                'Story','Plotline','The Beginning','Summary',
                'Content','Premise']

    # sometimes those possible names have 'Edit' latched onto the end due to user error on wikipedia
    # in that case, it will be 'PlotEdit' so it's easier to make another list that acccounts for that
    possibles_edit = [i + 'Edit' for i in possibles]

    #then merge those two lists together
    all_possibles = possibles + possibles_edit

    print("Starting the fetching plot process. This might take a while due to the size of the dataset, be patient...")

    title_plots = [] # initialize list to store plots

    """  i = wiki_titles[0]
    try:
            wik = wikipedia.WikipediaPage(i)
    except:
        wik = np.NaN
    
    try:
        # if no plot is found, then plot equals np.NaN
        plot_ = np.NaN
        # for all possible titles in all_possibles list
        for j in all_possibles:
            if wik.section(j) != None: # if that section does exist, i.e. it doesn't return 'None'
                print('found section')
                plot_ = wik.section(j).replace('\n','').replace("\'","")  #then that's what the plot is! Otherwise try the next one!
        title_plots.append({'primaryTitle': i, 'plot': plot_})
    except: # if the page didn't load from above, then plot equals np.NaN
        title_plots.append({'primaryTitle': i, 'plot': plot_}) """
    
    # fetch plots
    logs_helper = 0
    for i in wiki_titles:
    # loading the page once and save it as a variable, otherwise it will request the page every time
    # always do a try, except when pulling from the API, in case it gets confused by the title
        try:
            wik = wikipedia.WikipediaPage(i)
        except:
            wik = np.NaN

         # a new try, except for the plot
        try:
            # if no plot is found, then plot equals np.NaN
            plot_ = np.NaN
            # for all possible titles in all_possibles list
            for j in all_possibles:
                if wik.section(j) != None: # if that section does exist, i.e. it doesn't return 'None'
                    plot_ = wik.section(j).replace('\n','').replace("\'","")  #then that's what the plot is! Otherwise try the next one!
            title_plots.append({'primaryTitle': i, 'plot': plot_})

        except: # if the page didn't load from above, then plot equals np.NaN
            title_plots.append({'primaryTitle': i, 'plot': plot_})
        
        logs_helper = logs_helper+1
        if logs_helper % 50 == 0:
            print(f'checking ',logs_helper)
    print(title_plots[2])
        

    # create a df with the fetched plots
    plots_df = pd.DataFrame(title_plots)

    # Merge with the original DataFrame, using 'primaryTitle' as the key
    merged_df = imdb.merge(plots_df, on='primaryTitle', how='left')
    merged_df = merged_df.dropna(subset=['plot']) # dropping rows with null plots

    return merged_df


# ask if the user wants to run the plot fetching process
proceed = input("Do you want to proceed with fetching movie plots? (yes/no): ").strip().lower()

if proceed == 'yes':
    imdb = pd.read_csv("imdb.csv")
    imdb_with_plots = get_plots(imdb)
    print("Fetching completed.")
    print("IMDB with Plots: ", imdb_with_plots)
    imdb_with_plots.to_csv('imdb_with_plots.csv', index=False) 
else:
    print("Process cancelled.")
