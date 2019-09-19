from torch import nn
from torch.nn import functional as F
import torch

class BagModel_3d(nn.Module):
    '''
    BagModel_3d - not scalable, used with data represented in 3d
    Accepts 3d data tensor and n_instances array
    '''
    def __init__(self, prepNN, afterNN, aggregation_func):
        super().__init__()
        
        self.prepNN = prepNN
        self.afterNN = afterNN
        self.aggregation_func = aggregation_func

    @profile
    def forward(self, input, n_instances):
        #   n_instances.type(torch.long)
        
        NN_out = self.prepNN(input) # Forward all indices through neural network
        # print("DEBUG: NNout - {}".format(NN_out))
        
        output = torch.empty(size = (input.size(0), len(NN_out[0][0])), dtype = torch.double) # Pre-alocate tensor for output
        # print("DEBUG: empty output - {}".format(output))

        for i, n in enumerate(n_instances):
            output[i] = self.aggregation_func(NN_out[i, :n], dim = 0) # Aggregates only valid instances
            
        output = self.afterNN(output)
        
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
model = BagModel_3d(prepNN1, afterNN1, torch.mean)

#Data
data = torch.rand(size = (4, 4, 50)).double()
# n_instances = torch.Tensor([[4, 2, 3, 1]]).double()
n_instances = torch.Tensor([4, 2, 3, 1]).long()

output = model(data, n_instances)

print(output)
