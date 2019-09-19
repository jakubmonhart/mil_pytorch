import mill_python.create_dataset.create_bags as create_bags
import mill_python.src.mil_pytorch as mil
import mill_python.src.utils as utils
import mill_python.train.train as train
import torch
from torch.utils.data import DataLoader
from sklearn.datasets import make_classification

# Configurations
n_neurons1 = 5
n_neurons2 = 5
n_neurons3 = 5
learning_rate = 1e-3
weight_decay = 1e-4

# Create data
source_data, source_labels = make_classification(n_samples = 500, n_features = 5, n_informative = 5, n_redundant = 0, n_repeated = 0, n_classes = 10, class_sep = 1.0, n_clusters_per_class = 1)

data, ids, labels = create_bags.create_bags(source_data, source_labels, pos = 50, neg = 50, max_subbags = 7, max_instances = 7)
data = torch.Tensor(data).double()
ids = torch.Tensor(ids).long()
labels = torch.Tensor(labels).long()

# Convert from (1, 0) to (1, -1)
labels[labels == 0] = -1

# Dataset
dataset = mil.MilDataset(data, ids, labels, normalize = True)
train_ds, test_ds = mil.train_test_split(dataset, test_size = 0.2)

# Pre and after agg function
prepNN1 = torch.nn.Sequential(
    torch.nn.Linear(len(dataset.data[0]), n_neurons1, bias = True),
    torch.nn.ReLU(),
    torch.nn.Linear(n_neurons1, n_neurons2, bias = True),
    torch.nn.ReLU()
)

afterNN1 = torch.nn.Sequential(
    torch.nn.Identity()
)

prepNN2 = torch.nn.Sequential(
    torch.nn.Linear(n_neurons2, n_neurons3, bias = True),
    torch.nn.ReLU(),
)

afterNN2 = torch.nn.Sequential(
    torch.nn.Linear(n_neurons3, 1),
    torch.nn.Tanh()
)

# Model and loss function
model = torch.nn.Sequential(
    mil.BagModel(prepNN1, afterNN1, torch.mean),
    mil.BagModel(prepNN2, afterNN2, torch.mean)
).double()
criterion = mil.MyHingeLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

# Cross validation
loss = train.k_fold_cv(model, optimizer, criterion, train_ds, epochs = 300)
print('loss: {}'.format(loss.item()))
