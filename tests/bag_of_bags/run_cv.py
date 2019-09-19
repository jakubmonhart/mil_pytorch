import mil_pytorch.mil as mil
from mil_pytorch.utils import eval_utils, data_utils, train_utils, create_bags_simple, create_bags


import numpy
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.datasets import make_classification
import time



def create_model(input_len, n_neurons1, n_neurons2, n_neurons3):
    # Define neural networks for processing of data before and after aggregation
    prepNN1 = torch.nn.Sequential(
        torch.nn.Linear(input_len, n_neurons1, bias = True),
        torch.nn.ReLU(),
        torch.nn.Linear(n_neurons1, n_neurons2, bias = True),
        torch.nn.ReLU(),
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

    # Define model ,loss function and optimizer
    model = torch.nn.Sequential(
        mil.BagModel(prepNN1, afterNN1, torch.mean, device),
        mil.BagModel(prepNN2, afterNN2, torch.mean, device)
    ).double()
    criterion = mil.MyHingeLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    return model

# --- CONFIG ---

# Configurations
n_neurons1 = None
n_neurons2 = None
n_neurons3 = None
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 4000
pos = 100
neg = 100
class_sep = 1.2
n_features = 100
max_subbags = 5
max_instances = 5
batch_size = 0
patience = 1000
delta = 0

config = {}
config['epochs'] = epochs
config['pos'] = pos
config['neg'] = neg
config['class_sep'] = class_sep
config['n_features'] = n_features
config['max_subbags'] = max_subbags
config['max_instances'] = max_instances
config['batch_size'] = batch_size
config['patience'] = patience
config['delta'] = delta
config["n_neurons1"] = n_neurons1
config["n_neurons2"] = n_neurons2
config["n_neurons3"] = n_neurons3
config["learning_rate"] = learning_rate
config["weight_decay"] = weight_decay

# --- DATA ---

# Create data
source_data, source_labels = make_classification(n_samples = 20000, n_features = n_features, n_informative = n_features, n_redundant = 0, n_repeated = 0, n_classes = 10, class_sep = class_sep, n_clusters_per_class = 1)
data, ids, labels = create_bags.create_bags(source_data, source_labels, pos = pos, neg = neg, max_subbags = max_subbags, max_instances = max_instances)
print("Data shape:", data.shape, len(labels))

# Check if gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("INFO: Using device: {}".format(device))

# Move data to gpu (if available)
data = torch.Tensor(data).double().to(device)
ids = torch.Tensor(ids).long().to(device)
labels = torch.Tensor(labels).long().to(device)

# Convert labels from (1, 0) to (1, -1) for tanh
labels[labels == 0] = -1

# Create dataset and divide to train and test part
dataset = mil.MilDataset(data, ids, labels, normalize = True)

train_indices, valid_indices, test_indices = data_utils.data_split(dataset = dataset, valid_ratio = 0.0, test_ratio = 0.2, shuffle = True, stratify = True)

# print('DEBUG: train: {}, valid: {}, test: {}'.format(train_indices, valid_indices, test_indices))

train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(valid_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create dataloaders
if batch_size == 0:
    train_dl = DataLoader(dataset, sampler = train_sampler, batch_size = len(train_indices), collate_fn=mil.collate)
else:
    train_dl = DataLoader(dataset, sampler = train_sampler, batch_size = batch_size, collate_fn=mil.collate)

# valid_dl = DataLoader(dataset, sampler = valid_sampler, batch_size = len(valid_indices), collate_fn=mil.collate)
test_dl = DataLoader(dataset, sampler = test_sampler, batch_size = len(test_indices), collate_fn=mil.collate)


# ---- GRID SEARCH ---

n_neurons1_grid = [15]
n_neurons2_grid = [15]
n_neurons3_grid = [15]
# learning_rate_grid = numpy.logspace(-4, -2, num = 3)
# weight_decay_grid = numpy.logspace(-4, -2, num = 3)
learning_rate_grid = [1e-3]
weight_decay_grid = [1e-1]

for n_neurons1 in n_neurons1_grid:
    for n_neurons2 in n_neurons2_grid:
        for n_neurons3 in n_neurons3_grid:
            for learning_rate in learning_rate_grid:
                for weight_decay in weight_decay_grid:
                    config['n_neurons1'] = n_neurons1
                    config['n_neurons2'] = n_neurons2
                    config['n_neurons3'] = n_neurons3
                    config['learning_rate'] = learning_rate
                    config['weight_decay'] = weight_decay

                    print('INFO: Running cross validation with config:\n{}'.format(config))
 
                    # --- MODEL ---
                    model = create_model(len(dataset.data[0]), n_neurons1, n_neurons2, n_neurons3)
                    criterion = mil.MyHingeLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

                    # Move model to gpu if available
                    model = model.to(device)

                    avg_loss = train_utils.k_fold_cv(model = model, fit_fn = train_utils.train_model, criterion = criterion, optimizer = optimizer, dataset = dataset, train_indices = train_indices, epochs = epochs, patience = patience, delta = delta, device = device)

                    # Save log
                    train_utils.save_log(avg_loss, config)
