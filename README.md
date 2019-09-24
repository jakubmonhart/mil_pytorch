# mil_pytorch - multiple instance learning model implemented in pytorch
This library consists mainly of BagModel and MilDataset

```python
from mil_pytorch.mil import BagModel, MilDataset
```

BagModel is subclass of **torch.nn.Module** (see https://pytorch.org/docs/stable/nn.html#torch.nn.Module).  
MilDataset is subclass of **torch.utils.data.dataset** (see https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).  

For description of multiple instance learning problem see https://github.com/pevnak/Mill.jl#what-is-multiple-instance-learning-mil-problem.

## Usage
### Data
Each instance is feature vector with fixed lenght. A bag contains variable number of these instances. Each instance has an id specifying, which bag does it belong to (number of instances is then same as lenght of ids vector).
Initialize MilDataset passing it instances, ids and labels of bags ([1,-1] - [positive/negative]).
```python
dataset = MilDataset(instances, ids, labels)
```
You can access bags from this dataset by using index:
```python
instances, ids, label = dataset[index]
```
Or by iteration:
```python
for instances, ids, bag in dataset:
  ...
```

To use **torch.utils.data.DataLoader** (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) you need to use custom collate function.
```python
from torch.utils.data import DataLoader
import mil_pytorch.mil as mil

dataloader = DataLoader(dataset = dataset, batch_size = batch_size, collate_fn = mil.collate)
```

### Creating model
BagModel consists of user defined non-linear neural networks prepNN and afterNN and an aggregation function stacked between them.  
prepNN preserves number of feature vectors (instances) in input, aggregation function aggregates instances of each bag creating one feature vector for each bag. afterNN simply processes this output.

```python
# Define custom prepNN as single layer nn
prepNN = torch.nn.Sequential(
        torch.nn.Linear(input_len, 10, bias = True),
        torch.nn.ReLU(),
    )
   
# Define custom afterNN as single layer nn
afterNN = torch.nn.Sequential(
        torch.nn.Linear(10, 1, bias = True),
        torch.nn.Tanh(),
    )

# Define model with prepNN, afterNN and torch.mean
model = mil.BagModel(prepNN, afterNN, torch.mean)
```

Data for MIL problem can also be in the form of "bag of bags", where each bag consists of variable number of lower bags, which consists of variable number of instances. In this case ids is a matrix with number of columns equal to number of instances and number of rows equal to number of "bag-layers". Number of rows is 1 for "bag of instances" problem, 2 for "bag of bags" problem and so on.  
In the case of "bag of bags" problem, the neural network is created as sequence of two BagModels.
```python
# Define custom prepNN as single layer nn
prepNN1 = torch.nn.Sequential(
        torch.nn.Linear(input_len, 10, bias = True),
        torch.nn.ReLU(),
    )
   
# Define custom afterNN as single layer nn
afterNN1 = torch.nn.Sequential(
        torch.identity()
    )

prepNN2 = torch.nn.Sequential(
        torch.nn.Linear(5, 3, bias = True),
        torch.nn.ReLU()
    )

afterNN2 = torch.nn.Sequential(
        torch.nn.Linear(3, 1, bias = True),
        torch.nn.Tanh()
    )

# Define model with prepNN, afterNN and torch.mean
model = torch.nn.Sequential(
        mil.BagModel(prepNN1, afterNN1, torch.mean),
        mil.BagModel(prepNN2, afterNN2, torch.mean)
    )
```

Neural network created using this library has to consists only of instances of BagModel, it is not possible to combine mil.BagModel with other torch.nn modules due to correct pass of information about ids, but preNN and afterNN neural networks can be of arbitrary form as long as the length of output from prepNN matches the length of input of afterNN.
```python
# --- WRONG ---
model = torch.nn.Sequential(
     torch.nn.Linear(10, 10, bias = True),
     mil.BagModel(prepNN, afterNN, torch.mean),
     torch.nn.Linear(10, 1, bias = True)
     )
# --- WRONG ---
```
