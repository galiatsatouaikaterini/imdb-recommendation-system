import pandas as pd
import numpy as np

df = pd.read_csv('user_movie_ratings.csv')

# copy user dataset
df_users = df.copy()

# check for NaN values
print(df_users.isna().sum())

# ratings should be between 1-10 and only have 1 decimal
# round the rating column to one decimal
df_users['rating'] = df_users['rating'].round(1)
print(sorted(df_users['rating'].unique()))

print(df_users['rating'].value_counts())

# a lot of movies are rated 1.0 or 10.0 so I'll change 2000 values each
# 1.0
indices = df_users[df_users['rating'] == 1.0].index
indices_change = np.random.choice(indices, 2000, replace = False)
new_rating = np.random.choice([1.1, 1.2, 1.3], 2000)
df_users.loc[indices_change, 'rating'] = new_rating

# 10.0
indices2 = df_users[df_users['rating'] == 10.0].index
indices_change2 = np.random.choice(indices2, 2000, replace = False)
new_rating2 = np.random.choice([9.9, 9.8, 9.7], 2000)
df_users.loc[indices_change2, 'rating'] = new_rating2

# save the dataset
df_users.to_csv('user_movie_rating_cleaned.csv', index=False)