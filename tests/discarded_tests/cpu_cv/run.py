OMP_NUM_THREADS=1

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
    n_neurons1_grid = [5, 10, 15]
    n_neurons2_grid = [5, 10, 15]
    n_neurons3_grid = [5, 10, 15]
    weight_decay_grid = numpy.logspace(-4, -2, num = 3)
    learning_rate = 1e-3

    print("INFO: starting grid search")
    gs_start = time.time()

    for n_neurons1 in n_neurons1_grid:
        for n_neurons2 in n_neurons2_grid:
            for n_neurons3 in n_neurons3_grid:
                for weight_decay in weight_decay_grid:
                    run(n_neurons1, n_neurons2, n_neurons3, learning_rate, weight_decay)

    # run(10, 10, 10, 1e-3, 1e-4)

    print("INFO: total elapsed time: {}".format(time.time() - gs_start))

def run(n_neurons1, n_neurons2, n_neurons3, learning_rate, weight_decay):
    log = {}

    log["id"] = uuid.uuid4().hex
    log["description"] = "All nets are 1-layer, using mil_pytorch.py"
    log["config"] = {}
    log["config"]["n_neurons1"] = n_neurons1
    log["config"]["n_neurons2"] = n_neurons2
    log["config"]["n_neurons3"] = n_neurons3
    log["config"]["learning_rate"] = learning_rate
    log["config"]["weight_decay"] = weight_decay.item()

    # Create data
    source_data, source_labels = make_classification(n_samples = 100, n_features = 5, n_informative = 5, n_redundant = 0, n_repeated = 0, n_classes = 10, class_sep = 1.0, n_clusters_per_class = 1)

    data, ids, labels = create_bags.create_bags(source_data, source_labels, pos = 30, neg = 30, max_subbags = 5, max_instances = 5)

    print("Data shape:", data.shape, len(labels))
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
    print("INFO: Running cross validation with configuration")
    print(log["config"])
    start = time.time()
    loss = train.k_fold_cv(model, optimizer, criterion, train_ds, epochs = 2000)
    log["loss"] = loss.item()
    log["time"] = time.time() - start
    print("loss: {}".format(log["loss"]))
    print('INFO: Finished training - elapsed time: {}'.format(log["time"]))

    with open('yaml_logs/{}.yml'.format(log['id']), 'w') as outfile:
        yaml.dump(log, outfile, default_flow_style=False)
        print(yaml.dump(log))

    print("________________________________________")
    

if __name__ == '__main__':
    grid_search()
