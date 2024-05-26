import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import time
from tqdm import tqdm

# ask if the user wants to run the movielens
proceed = input("Do you want to proceed with movielens or imdb? (movielens/imdb): ").strip().lower()

if proceed == 'movielens':
    imdb = pd.read_csv("movie_small.csv")
    # we need a rating dataframe with userid, movieid, rating
    ratings = pd.read_csv("rating_small.csv")
    # sorting ratings by timestamp for the splitting
    ratings = ratings.sort_values('timestamp')
    print(ratings.head())

elif proceed == 'imdb':
    imdb = pd.read_csv("imdb_with_plots.csv")
    ratings = pd.read_csv("user_movie_rating_cleaned.csv")
    # dropping unecessary columns
    imdb.drop(columns = ['titleType', 'originalTitle', 'startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'directors', 'plot' ], axis=1,  inplace=True)
    imdb.rename(columns = {'tconst' : 'movieId', 'primaryTitle': 'title'}, inplace=True)
    ratings.rename(columns = {'userID':'userId', 'movieID':'movieId'}, inplace=True)
    ratings['userId'] = ratings['userId'].str.replace('user_', '').astype(int)
else:
    print("Invalid input.")


# ratings = ratings1.sample(n=40000, random_state=42)
# imdb = imdb1.sample(n=40000, random_state=42)

unique_users = ratings['userId'].nunique()

print(f'There are {unique_users} unique users in the dataset.')

def parse_genres(genres_str):
    if proceed == 'movielens':
    # split the genres string at each '|' and create a list
        return genres_str.split('|') # this method directly converts the string to a list where each element is a genre previously separated by '|'
    elif proceed == 'imdb':
        return genres_str.split(',')
    
imdb['genres'] = imdb['genres'].apply(parse_genres)
print(imdb.head())


# ----------------------------- Encoding --------------------------------------

# using LabelEncoder from sklearn to convert categorical user and movie ids into numerical format for model training
user_encoder = LabelEncoder()
title_encoder = LabelEncoder()

user_ids = user_encoder.fit_transform(ratings.userId)
title_ids = title_encoder.fit_transform(ratings.movieId)

# --------------------------- Splitting Dataset ------------------------------

# spliting the data into training (80%) and validation (20%) sets based on the user and movie ids
num_train = int(len(user_ids) * 0.8)

train_user_ids = user_ids[:num_train] # training (80%)
train_title_ids = title_ids[:num_train]
train_ratings = ratings.rating.values[:num_train]

val_user_ids = user_ids[num_train:] # validation (20%)
val_title_ids = title_ids[num_train:]
val_ratings = ratings.rating.values[num_train:]

# normalisation of ratings
if proceed == 'movielens':
    train_ratings /= 5 # normalises the ratings to be between 0 and 1 by dividing by the maximum possible rating (assumed to be 5)
    val_ratings /= 5 
elif proceed == 'imdb':
    train_ratings /= 10 # normalises the ratings to be between 0 and 1 by dividing by the maximum possible rating (assumed to be 5)
    val_ratings /= 10 

# ------------------------------- NCF Model ---------------------------------

class NCFModel(nn.Module): # Neural Collaborative Filtering (NCF)
    def __init__(self, num_users, num_items, embedding_dim=64, mlp_layers=[128, 64]):

        super(NCFModel, self).__init__()
        # GMF part
        # user and item embedding layers: users and items are represented using dense vector embeddings, which are learned during training
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)  # these embeddings capture latent features of users and items
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP part
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential()
        previous_layer_size = embedding_dim * 2  # input size to the first MLP layer

        for layer_size in mlp_layers:
            self.mlp.add_module('dense_{}'.format(layer_size), nn.Linear(previous_layer_size, layer_size))
            self.mlp.add_module('relu_{}'.format(layer_size), nn.ReLU())
            self.mlp.add_module('dropout_{}'.format(layer_size), nn.Dropout(0.2))
            previous_layer_size = layer_size

        # final layer
        self.output = nn.Linear(embedding_dim + mlp_layers[-1], 1)

    def forward(self, user_indices, item_indices):
        # GMF part
        user_embedding_gmf = self.user_embedding_gmf(user_indices)
        item_embedding_gmf = self.item_embedding_gmf(item_indices)
        gmf_output = user_embedding_gmf * item_embedding_gmf

        # MLP part
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat((user_embedding_mlp, item_embedding_mlp), dim=1)
        mlp_output = self.mlp(mlp_input)

        # concatenate GMF and MLP outputs
        final_features = torch.cat((gmf_output, mlp_output), dim=1)
        output = torch.sigmoid(self.output(final_features))

        return output


# ---------------------------- Model Compilation & Training Setup ---------------------------

num_users = unique_users
num_items = imdb.shape[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
model = NCFModel(num_users, num_items)
model.to(device)  # move the model to the appropriate device

criterion = nn.MSELoss()  # mean squared error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


# converting the numpy arrays to torch tensors
train_user_ids = torch.tensor(train_user_ids).long()
train_movie_ids = torch.tensor(train_title_ids).long()
train_ratings = torch.tensor(train_ratings).float()
val_user_ids = torch.tensor(val_user_ids).long()
val_movie_ids = torch.tensor(val_title_ids).long()
val_ratings = torch.tensor(val_ratings).float()

# creating the dataloaders
train_dataset = TensorDataset(train_user_ids, train_movie_ids, train_ratings)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

val_dataset = TensorDataset(val_user_ids, val_movie_ids, val_ratings)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    total_samples = len(data_loader.dataset)  # Total samples for averaging loss

    # Initialize tqdm progress bar
    progress_bar = tqdm(data_loader, desc='Training', total=len(data_loader))

    for data in progress_bar:
        user, item, ratings = data
        user, item, ratings = user.to(device), item.to(device), ratings.to(device)

        optimizer.zero_grad()
        outputs = model(user, item).squeeze()
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()

        # Update running loss for averaging later
        running_loss += loss.item() * user.size(0)

        # Update the progress bar with the current batch loss
        progress_bar.set_postfix(loss=(running_loss / total_samples))

    average_loss = running_loss / total_samples
    progress_bar.close()  # Ensure the progress bar is properly closed after completion
    return average_loss


# defining the validation function
def validate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            user, item, ratings = data
            user, item, ratings = user.to(device), item.to(device), ratings.to(device)

            outputs = model(user, item).squeeze()
            loss = criterion(outputs, ratings)

            running_loss += loss.item() * user.size(0)
    return running_loss / len(data_loader.dataset)





# -------------------------------- Training --------------------------------

epochs = 50
best_val_loss = np.inf
patience = 1
trigger_times = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs): # training loop
    start_time = time.time()

    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_model(model, val_loader, criterion)
    #print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    end_time = time.time()
    epoch_duration = end_time - start_time
    epoch_duration_str = time.strftime("%H:%M:%S", time.gmtime(epoch_duration))

    # calculating the number of samples processed
    num_train_samples = len(train_loader.dataset)
    num_val_samples = len(val_loader.dataset)

    print(f'Epoch {epoch+1}/{epochs}')
    print(f'{num_train_samples}/{num_train_samples} [==============================] - {epoch_duration_str} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')

    # early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        print("Validation loss decreased, saving model...")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping! Training stopped at epoch {epoch+1}')
            break


def ndcg_at_k(sorted_ratings, k=10): # calculating Normalized Discounted Cumulative Gain at K
    # ensuring that there are at least `k` ratings to calculate NDCG properly
    if sorted_ratings.numel() < k:
        # print("Inside the insufficient if...")
        return 0.0  # not enough ratings to calculate NDCG for top-k
    
    sorted_ratings = sorted_ratings[:k] # ensuring that sorted_ratings does not exceed k items
    
    # proceed with calculation if there are enough ratings
    discount = torch.log2(torch.arange(2, k+2, dtype=torch.float, device=sorted_ratings.device))
    gains = (2**sorted_ratings - 1) / discount
    dcg = gains.sum()

    # calculate ideal DCG (iDCG) using the best possible ordering of actual ratings
    ideal_ratings = torch.sort(sorted_ratings, descending=True).values[:k]
    ideal_gains = (2**ideal_ratings - 1) / discount
    idcg = ideal_gains.sum()

    return (dcg / idcg).item() if idcg > 0 else 0 # return NDCG value

def hit_ratio(ranklist, gt_item): # calculating Hit Ratio
    # checking if the ground-truth item is in the recommended items
    return 1 if gt_item in ranklist else 0


# --------------------------------- Prediction -----------------------------------

def evaluate_prediction(predictions, val_user_ids, val_movie_ids, val_ratings, k=10): # evaluating the performance (NDCG and Hit Ratio) of top-K recommendation
       # predictions: torch.Tensor user-item predictions
       # val_user_ids, val_movie_ids, val_ratings: torch.Tensors
    
    ndcgs = []
    hit_ratios = []
    unique_users = val_user_ids.unique()  # get unique users

    for target_user in unique_users:
        # get movie ids and ratings associated with the target user
        user_mask = (val_user_ids == target_user)
        target_val_movie_ids = val_movie_ids[user_mask]
        target_val_ratings = val_ratings[user_mask]
        user_predictions = predictions[user_mask]

        # sort ratings based on predictions
        _, indices = torch.sort(user_predictions, descending=True)
        sorted_ratings = target_val_ratings[indices]
        target_movie_ids_sorted = target_val_movie_ids[indices]

        # ground truth item (the one actually interacted with)
        gt_item = target_val_movie_ids[val_ratings[user_mask].argmax()]

        # computing NDCG and Hit Ratio for this user
        ndcg = ndcg_at_k(sorted_ratings, k=k)
        hr = hit_ratio(target_movie_ids_sorted[:k], gt_item)

        ndcgs.append(ndcg)
        hit_ratios.append(hr)

    average_ndcg = torch.tensor(ndcgs, dtype=torch.float).mean().item()  # calculating mean NDCG
    average_hit_ratio = torch.tensor(hit_ratios, dtype=torch.float).mean().item()  # calculating mean Hit Ratio
    
    # average_ndcg: float, average NDCG for each user
    # average_hit_ratio: float, average Hit Ratio for each user

    return average_ndcg, average_hit_ratio, ndcgs, hit_ratios

# example usage
model.eval()  # set the model to evaluation mode
with torch.no_grad():  # context manager that disables gradient calculation
    predictions = model(val_user_ids, val_movie_ids)
    average_ndcg, average_hit_ratio, ndcgs, hit_ratios = evaluate_prediction(predictions.squeeze(), val_user_ids, val_movie_ids, val_ratings, k=10)
    print(f'Average NDCG: {average_ndcg:.4f}')
    print(f'Average Hit Ratio: {average_hit_ratio:.4f}')
    #print(f' Hit Ratios: {hit_ratios}')
    #print(f'NDCG: {ndcgs}')