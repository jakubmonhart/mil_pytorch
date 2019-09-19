from torch import nn
from torch.nn import functional as F
import torch
from sklearn import metrics
import time
from sklearn.datasets import make_classification
from mil_pytorch.create_dataset.create_bags_simple import create_bags_simple
import numpy
from sklearn import model_selection
import mil_pytorch.src.mil_pytorch as mil



class BagModel(nn.Module):
    '''
    BagModel used with data represented as sequence of instances along with array specifiing number of instances
    Accepts data tensor and n_instances array
    '''
    def __init__(self, prepNN, afterNN, aggregation_func, device = 'cpu'):
        super().__init__()
        
        self.device = device
    
        self.prepNN = prepNN
        self.afterNN = afterNN
        self.aggregation_func = aggregation_func
    
    def forward(self, input, n_instances):    
        NN_out = self.prepNN(input) # Forward all instances through neural network
        output = torch.empty(n_instances.size(0), len(NN_out[0])).to(device)
        
        for i in range(len(n_instances)):
            start = torch.sum(n_instances[:i])
            end = start + n_instances[i]
            output[i] = self.aggregation_func(NN_out[start:end], dim = 0)
        
        output = self.afterNN(output.double())
        
        return output



def ids2n_instances(ids):
    unique, inverse, counts = torch.unique(ids, sorted = True, return_inverse = True, return_counts = True)
    idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
    bags = unique[idx]
    counts = counts[idx]
    
    return counts



def data_split(data, labels, n_instances, test_size = 0.2):
    data = numpy.array(data)
    labels = numpy.array(labels)
    
    y, y_t, n_instances, n_instances_t = model_selection.train_test_split(labels, n_instances, test_size = test_size, shuffle = False)
    
    print('DEBUG: sum(n_instances): {}'.format(sum(n_instances.astype(int))))
    x = data[:sum(n_instances.astype(int))]
    x_t = data[sum(n_instances.astype(int)):]

    y = torch.from_numpy(y).float()
    x = torch.from_numpy(x)
    x_t = torch.from_numpy(x_t)
    y_t = torch.from_numpy(y_t).float()
    n_instances = torch.from_numpy(n_instances).long()
    n_instances_t = torch.from_numpy(n_instances_t).long()
    
    return x, x_t, y, y_t, n_instances, n_instances_t



def accuracy(pred, target, threshold = 0):
    '''
    '''

    pred = pred.detach().numpy()
    target = target.detach().numpy()

    pred[pred >= threshold] = 1
    pred[pred < threshold] = -1

    return numpy.sum(target == pred)/target.shape[0]



def eer(pred, labels):
    fpr, tpr, threshold = metrics.roc_curve(labels.detach(), pred.detach(), pos_label=1)
    fnr = 1 - tpr
    EER_fpr = fpr[numpy.nanargmin(numpy.absolute((fnr - fpr)))]
    EER_fnr = fnr[numpy.nanargmin(numpy.absolute((fnr - fpr)))]
    return EER_fpr, EER_fnr



# Configuration
n_neurons1 = 15
learning_rate = 1e-3
weight_decay = 1e-3
epochs = 100
pos = 500
neg = 500
n_features = 100
class_sep = 1.0
max_instances = 20

# Pre and after agg function
prepNN1 = torch.nn.Sequential(
    torch.nn.Linear(n_features, n_neurons1, bias = True),
    torch.nn.ReLU(),
).double()

afterNN1 = torch.nn.Sequential(
    torch.nn.Linear(n_neurons1, 1, bias = True),
    torch.nn.Tanh()
).double()

# Create data
source_data, source_labels = make_classification(n_samples = 5000, n_features = n_features, n_informative = n_features, n_redundant = 0, n_repeated = 0, n_classes = 10, class_sep = class_sep, n_clusters_per_class = 1)
data, ids, labels = create_bags_simple(source_data, source_labels, pos = pos, neg = neg, max_instances = max_instances)
n_instances = ids2n_instances(torch.Tensor(ids))

print("Data shape:", data.shape, len(labels))

# Convert from (1, 0) to (1, -1)
labels[labels == 0] = -1

# Check if gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("INFO: Using device: {}".format(device))

# Split data on train and test sets
labels = numpy.array(labels)
data = numpy.array(data)
n_instances = numpy.array(n_instances.float())
x, x_t, y, y_t, n_instances, n_instances_t = data_split(data, labels, n_instances)
print('x: {} x_t: {}'.format(x.shape, x_t.shape))
print('y: {} y_t: {}'.format(y.shape, y_t.shape))
print('n_instances: {} n_instances_t: {}'.format(n_instances.shape, n_instances_t.shape))

# # Move data to gpu (if available)
x = x.double().to(device)
y = y.long().to(device)
n_instances = n_instances.to(device)

x_t = x_t.double().to(device)
y_t = y_t.long().to(device)
n_instances_t = n_instances_t.to(device)

# Init model
model = BagModel(prepNN1, afterNN1, torch.mean, device = device)
model = model.double()

# Move model to gpu if available
model = model.to(device)

# Criterion and optimizer
criterion = mil.MyHingeLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

# Fit
start = time.time()

print_loss = True
x = x.double()
# n_instances = torch.Tensor(n_instances).long()



for epoch in range(epochs):
    pred = model(x, n_instances)
    loss = criterion(pred[:,0], y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every ** epochs
    if print_loss and ((epoch+1)%50 == 0):
        print('step: {:3d} loss: {}'.format(epoch+1, loss))

print('INFO | Finished training - elapsed time: {}'.format(time.time() - start))

# Train set accuracy
pred = model(x, n_instances) 
loss = criterion(pred[:,0], y)
pred = pred.cpu()
# y = y.to('cpu')
eer_fpr, eer_fnr = eer(pred[:,0], y)
print('loss_train: {}'.format(loss))
print('acc_train: {}'.format(accuracy(pred[:,0], y)))
print('eer_fpr_train: {}'.format(eer_fpr))
print('eer_fnr_train: {}'.format(eer_fnr))

# Test set accuracy
x_t = x_t.double()
pred = model(x_t, n_instances_t)
loss = criterion(pred[:,0], y_t)
pred = pred.cpu()
# y_t = y_t.to('cpu')
eer_fpr, eer_fnr = eer(pred[:,0], y_t)
print('loss_test: {}'.format(loss))
print('acc_test: {}'.format(accuracy(pred[:,0], y_t)))
print('eer_fpr_test: {}'.format(eer_fpr))
print('eer_fnr_test: {}'.format(eer_fnr))