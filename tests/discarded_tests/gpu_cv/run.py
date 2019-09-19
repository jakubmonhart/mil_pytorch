import mill_python.create_dataset.create_bags as create_bags
import mill_python.src.mil_pytorch as mil
import mill_python.src.utils as utils
import mill_python.train.train as train
import torch
from torch.utils.data import DataLoader
from sklearn.datasets import make_classification
import yaml
import uuid
import numpy
import time


def grid_search():
    # Cross validation
    n_neurons1_grid = range(2,7)
    # n_neurons2_grid = range(2,7)
    # n_neurons3_grid = range(2,7) 
    # learning_rate_grid = numpy.logspace(-6, -2, num = 5)
    learning_rate = 1e-3
    weight_decay_grid = numpy.logspace(-4, -2, num = 3)

    print("Starting grid search")
    gs_start = time.time()

    for n_neurons1 in n_neurons1_grid:
        for weight_decay in weight_decay_grid:
            run(n_neurons1, weight_decay)

    print("Total elapsed time: {}".format(time.time() - gs_start))

def run(n_neurons1, weight_decay):
    log = {}

    # # Configurations
    # n_neurons1 = 5
    # n_neurons2 = 5
    # n_neurons3 = 5
    # learning_rate = 1e-3
    # weight_decay = 1e-4

    log["id"] = uuid.uuid4().hex
    log["description"] = "..."
    log["config"] = {}
    log["config"]["n_neurons1"] = n_neurons1
    # log["config"]["n_neurons2"] = n_neurons2
    # log["config"]["n_neurons3"] = n_neurons3
    log["config"]["learning_rate"] = learning_rate.item()
    log["config"]["weight_decay"] = weight_decay.item()

    # Create data
    source_data, source_labels = make_classification(n_samples = 500, n_features = 5, n_informative = 5, n_redundant = 0, n_repeated = 0, n_classes = 10, class_sep = 1.0, n_clusters_per_class = 1)

    data, ids, labels = create_bags.create_bags(source_data, source_labels, pos = 30, neg = 30, max_subbags = 5, max_instances = 5)

    # Move data to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    data = torch.Tensor(data).double().to(device)
    ids = torch.Tensor(ids).long().to(devide)
    labels = torch.Tensor(labels).long().to(device)

    # Convert from (1, 0) to (1, -1)
    labels[labels == 0] = -1

    # Dataset
    dataset = mil.MilDataset(data, ids, labels, normalize = True)
    train_ds, test_ds = mil.train_test_split(dataset, test_size = 0.2)

    # Pre and after agg function
    prepNN1 = torch.nn.Sequential(
        torch.nn.Linear(len(dataset.data[0]), n_neurons1, bias = True),
        torch.nn.ReLU(),
    )

    afterNN1 = torch.nn.Sequential(
        torch.nn.Identity()
    )

    prepNN2 = torch.nn.Sequential(
        torch.nn.Linear(n_neurons1, n_neurons3, bias = True),
        torch.nn.ReLU(),
    )

    afterNN2 = torch.nn.Sequential(
        torch.nn.Linear(n_neurons1, 1),
        torch.nn.Tanh()
    )

    # Model and loss function
    model = torch.nn.Sequential(
        mil.BagModel(prepNN1, afterNN1, torch.mean),
        mil.BagModel(prepNN2, afterNN2, torch.mean)
    ).double()
    criterion = mil.MyHingeLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    # Move model to gpu
    model = model.to(device)

    # Cross validation
    print("Running cross validation with configuration")
    print(log["config"])
    
    # Move everything to gpu

    # Training
    start = time.time()

    loss = train.k_fold_cv(model, optimizer, criterion, train_ds, epochs = 2000)
    log["loss"] = loss.item()
    log["time"] = time.time() - start

    with open('yaml_logs/{}.yml'.format(log['id']), 'w') as outfile:
        yaml.dump(log, outfile, default_flow_style=False)
        print(yaml.dump(log))

    print("________________________________________")
    

if __name__ == '__main__':
    grid_search()
