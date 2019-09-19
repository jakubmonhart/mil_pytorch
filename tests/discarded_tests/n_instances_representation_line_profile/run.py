import torch
import torch.nn as nn

class BagModel(nn.Module):
    '''
    BagModel used with data represented as sequence of instances along with array specifiing number of instances
    Accepts data tensor and n_instances array
    '''
    def __init__(self, prepNN, afterNN, aggregation_func):
        super().__init__()

        # TODO - Check for valid dimensions of prepNN and afterNN

        self.prepNN = prepNN
        self.aggregation_func = aggregation_func
        self.afterNN = afterNN
    
    @profile
    def forward(self, input, n_instances):    
        NN_out = self.prepNN(input) # Forward all instances through neural network
        output = torch.empty(n_instances.size(0), len(NN_out[0])) # Allocates empty tensor for aggregated output
        
        for i in range(len(n_instances)):
            start = torch.sum(n_instances[:i])
            end = start + n_instances[i]
            output[i] = self.aggregation_func(NN_out[start:end], dim = 0)
        
        output = self.afterNN(output.double())
        
        return output


# Pre and after agg function
prepNN1 = torch.nn.Sequential(
    torch.nn.Linear(50, 5, bias = True),
    torch.nn.ReLU(),
).double()

afterNN1 = torch.nn.Sequential(
    torch.nn.Linear(5, 1, bias = True),
    torch.nn.Tanh()
).double()

# Init model
model = BagModel(prepNN1, afterNN1, torch.mean)

#Data
data = torch.rand((10,50)).double()
n_instances = torch.Tensor([4, 2, 3, 1]).long()

output = model(data, n_instances)

print(output)