
import torch
import torch.nn as nn
import torch.optim as optim

 # NCF uses neural networks to model more complex, non-linear interactions

class NCF(nn.Module): # Neural Collaborative Filtering (NCF)
    def __init__(self, num_users, num_items, embedding_dim=8, hidden_layers=[64, 32, 16, 8]):
        super(NCF, self).__init__()

        # user and item embedding layers: users and items are represented using dense vector embeddings, which are learned during training, these embeddings capture latent features of users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        mlp_layers = [] # defining MLP layers
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_layers: # the embeddings pass through multiple hidden layers that learn high-level patterns and non-linear interactions between users and items
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)

        self.output_layer = nn.Linear(input_dim, 1) # output layer: the network outputs a score that represents a user's predicted preference for a particular item


    def forward(self, user_id, item_id):
        # getting the embeddings for the user and the item
        user_embedding = self.user_embedding(user_id) # NCF consists of two embedding layers: one for users and one for items, which are passed through an MLP
        item_embedding = self.item_embedding(item_id)

        x = torch.cat([user_embedding, item_embedding], dim=-1) # concatenating the user and item embeddings

        x = self.mlp(x) # passing through MLP

        x = self.output_layer(x)  # final output layer for prediction
        return torch.sigmoid(x)


# test
num_users = 1000
num_items = 500
embedding_dim = 16
model = NCF(num_users, num_items, embedding_dim)

# defining the loss function and optimizer
criterion = nn.BCELoss() # binary cross-entropy loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)

# input data (dummy user-item interactions) of cpurse here we have to put our data 
user_input = torch.tensor([1, 5, 9])
item_input = torch.tensor([10, 20, 30])
ratings = torch.tensor([1.0, 0.0, 1.0])

# training loop (simplified)
optimizer.zero_grad()
predictions = model(user_input, item_input)
loss = criterion(predictions.squeeze(), ratings)
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")
