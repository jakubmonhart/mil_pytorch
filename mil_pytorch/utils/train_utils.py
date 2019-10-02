import torch
import numpy
from torch.utils.data import DataLoader, SubsetRandomSampler
import mil_pytorch.mil as mil
import math
from mil_pytorch.utils import eval_utils, data_utils, create_bags_simple

import yaml
import uuid
import time

def fit(model, optimizer, criterion, train_dl, epochs = 1000, print_loss = False):
    '''
    Fit function returning error on validation data
    '''

    for epoch in range(epochs):
        for data, info, labels in train_dl:
            pred = model((data, info))
            loss = criterion(pred[:,0], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        # Print loss every so epochs
        if print_loss and ((epoch+1)%50 == 0):
            print('step: {:3d} loss: {}'.format(epoch+1, loss))


def train_model(model, criterion, optimizer, train_dl, valid_dl, epochs, patience, delta, device = 'cpu'):
    start = time.time()
    print('TRAINING:')

    # Train model
    train_losses = torch.Tensor(0).to(device)
    valid_losses = torch.Tensor(0).to(device)

    early_stopping = EarlyStopping(patience = patience , delta = delta)

    early_stop = False
    epoch = 0

    while (epoch < epochs) and not early_stop:
        
        # Optimization
        for data, info, labels in train_dl:
            pred = model((data, info))
            loss = criterion(pred[:,0], labels)

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses = torch.cat((train_losses, loss.float()))

        # Validation
        for data, info, labels in valid_dl:

            pred = model((data, info))
            loss = criterion(pred[:,0], labels)

            valid_losses = torch.cat((valid_losses, loss.float()))

        train_loss = torch.mean(train_losses, dim = 0, keepdim = True)
        valid_loss = torch.mean(valid_losses, dim = 0, keepdim = True)

        train_losses = torch.Tensor(0).to(device)
        valid_losses = torch.Tensor(0).to(device)

        # Early stopping
        stop = early_stopping(valid_loss, model)

        if stop:
            print('INFO: Early stopped - val_loss_min: {}'.format(early_stopping.val_loss_min.item()))
            # Load best parameters
            model.load_state_dict(early_stopping.saved_state_dict)
            early_stop = True

        # Print message
        if (epoch+1)%100 == 0:
            print('[{}/{}] | train_loss: {} | valid_loss: {} '.format(epoch+1, epochs, train_loss.item(), valid_loss.item()))

        epoch += 1
   
    print('INFO: Finished training - elapsed time: {}'.format(time.time() - start))

    return early_stopping.val_loss_min



def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a = math.sqrt(5))

        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(m.bias, -bound, bound)



def k_fold_cv(model, fit_fn, criterion, optimizer, dataset, train_indices, epochs, patience, delta, n_folds = 5, print_loss = True, device = 'cpu'):

    losses = torch.Tensor(0).to(device)

    # iterate over parts
    n_samples = len(train_indices)
    fold_sizes = numpy.full(n_folds, n_samples // n_folds, dtype=numpy.int)
    fold_sizes[:n_samples % n_folds] += 1
    current = 0

    for i, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        cv_valid_indices = train_indices[start:stop]
        cv_train_indices = numpy.concatenate((train_indices[:start], train_indices[stop:]))
        current = stop

        # Use subsetrandomsampler as sampler .. instead of shuffle, thanks to it, you will use only one dataset
        cv_train_sampler = SubsetRandomSampler(indices = cv_train_indices)
        cv_valid_sampler = SubsetRandomSampler(indices = cv_valid_indices)

        train_dl = DataLoader(dataset = dataset, batch_size = len(cv_train_indices), sampler = cv_train_sampler, collate_fn = mil.collate)
        valid_dl = DataLoader(dataset = dataset, batch_size = len(cv_valid_indices), sampler = cv_valid_sampler, collate_fn = mil.collate)

        # Reset model parameters
        model.apply(weight_init)

        # Fit and save error
        print("K-fold CV: [{}/{}]".format(i+1,len(fold_sizes)))
        min_loss = fit_fn(model, criterion, optimizer, train_dl, valid_dl, epochs, patience, delta, device)

        losses = torch.cat((losses, min_loss))

    # Return mean of errors
    return torch.mean(torch.Tensor(losses))


class EarlyStopping():
    def __init__(self, patience = 10, delta = 0):
        self.patience = patience
        self.delta = delta
        self.val_loss_min = None
        self.saved_state_dict = None
        self.counter = 0

    def __call__(self, val_loss, model):
        if self.val_loss_min is None:
            self.val_loss_min = val_loss
            self.saved_state_dict = model.state_dict()
            return False

        change = (self.val_loss_min - val_loss) / self.val_loss_min

        if change >= self.delta:
            self.counter = 0
            self.val_loss_min = val_loss
            self.saved_state_dict = model.state_dict()
            return False
        else:
            self.counter += 1

            if self.counter > self.patience:
                return True
            else:
                return False

def save_log(loss, config):
    log = {}

    log['id'] = uuid.uuid4().hex
    log['description'] = ''

    log['loss'] = loss.item()
    log['config'] = config

    with open('yaml_logs/{}.yml'.format(log['id']), 'w') as outfile:
        yaml.dump(log, outfile, default_flow_style=False)
