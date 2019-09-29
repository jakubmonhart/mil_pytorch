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
Each instance is feature vector with fixed lenght. A bag contains variable number of these instances. Each instance has an id specifying, which bag does it belong to. Ids of instances are stored in vector with length equal to number of instances.

Initialize MilDataset passing it instances, ids and labels of bags.

```python
import torch
import mil_pytorch.mil as mil

# Create 4 instances divided to 2 bags in 3:1 ratio. First bag has positive label, second bag has negative label
instances = torch.tensor([[1.0, 1.0, 1.0, 1.0],
				   		  [2.0, 2.0, 2.0, 2.0],
						  [3.0, 3.0, 3.0, 3.0],
						  [4.0, 4.0, 4.0, 4.0]], dtype = torch.double)
ids = torch.tensor([0, 0, 0, 1], dtype = torch.long)
labels = torch.tensor([1, -1], dtype = torch.long)

# Initialize MilDataset using created data
dataset = mil.MilDataset(instances, ids, labels)
```

You can access bags from this dataset using index ...  

```python
instances, ids, label = dataset[index]
```

... or iteration.

```python
for instances, ids, bag in dataset:
  ...
```

To use **torch.utils.data.DataLoader** (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) you need to use custom collate function **mil.collate**.

```python
from torch.utils.data import DataLoader
import mil_pytorch.mil as mil

dataloader = DataLoader(dataset = dataset, batch_size = batch_size, collate_fn = mil.collate)
```

### Creating model
BagModel consists of user defined non-linear neural networks prepNN and afterNN and an aggregation function stacked between them.  
PrepNN preserves number of feature vectors (instances) in input, aggregation function aggregates instances of each bag creating one feature vector per bag. This output is then forwarded through afterNN.

```python
# Define custom prepNN
prepNN = torch.nn.Sequential(
        torch.nn.Linear(input_len, 10, bias = True),
        torch.nn.ReLU(),
    )
   
# Define custom afterNN
afterNN = torch.nn.Sequential(
        torch.nn.Linear(10, 1, bias = True),
        torch.nn.Tanh(),
    )

# Define model with prepNN, afterNN and torch.mean as aggregation function
model = mil.BagModel(prepNN, afterNN, torch.mean)
```

### Bag of bags

Data for MIL problem can also be in the form of "bag of bags", where each bag consists of variable number of lower bags, which consists of variable number of instances. In this case ids is a matrix with number of columns equal to number of instances and number of rows equal to number of "bag-layers". For example number of rows is 1 for "bag of instances" problem, 2 for "bag of bags" problem and so on.

Matrix of ids is sorted in such a way, that last row specifies ids of instances, the row above specifies ids of sub-bags and so on to top.

Data for bag of bags problem would look like this:

```python
import torch
import mil_pytorch.mil as mil

# Create 8 instances divided to 4 lower-bags in 1:3:2:2 ratio. Lower-bags are divided into 2 bags in ratio 2:2 First bag has positive label, second bag has negative label
instances = torch.tensor([[1.0, 1.0, 1.0, 1.0],
						  [2.0, 2.0, 2.0, 2.0],
						  [3.0, 3.0, 3.0, 3.0],
						  [4.0, 4.0, 4.0, 4.0],
						  [5.0, 5.0, 5.0, 5.0],
						  [6.0, 6.0, 6.0, 6.0],
						  [7.0, 7.0, 7.0, 7.0],
						  [8.0, 8.0, 8.0, 8.0]], dtype = torch.double)
						  
ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1],
					[0, 1, 1, 1, 2, 2, 3, 3]], dtype = torch.long)
					
labels = torch.tensor([1, -1], dtype = torch.long)

# Initialize MilDataset using created data
dataset = mil.MilDataset(instances, ids, labels)
```



In the case of "bag of bags" problem, the neural network is created as sequence of two BagModels.

```python
# Define prepNNs and afterNNs
prepNN1 = torch.nn.Sequential(
        torch.nn.Linear(input_len, 10, bias = True),
        torch.nn.ReLU(),
    )
   
afterNN1 = torch.nn.Sequential( 
        torch.identity() # In this case afterNN1 and prepNN2 are interchangeable
    )

prepNN2 = torch.nn.Sequential(
        torch.nn.Linear(5, 3, bias = True),
        torch.nn.ReLU()
    )

afterNN2 = torch.nn.Sequential(
        torch.nn.Linear(3, 1, bias = True),
        torch.nn.Tanh()
    )

# Define model with prepNN, afterNN and torch.mean as aggregation function
model = torch.nn.Sequential(
        mil.BagModel(prepNN1, afterNN1, torch.mean),
        mil.BagModel(prepNN2, afterNN2, torch.mean)
    )
```

### 3d data representation
If using data in form of bag of instances (the simplest case), it's possible to use **mil.BagModel_3d** and **mil.MilDataset_3d** instead of the ones used above. In some cases, this can lead to speed up of forward function of the model. This method is however more memory consuming, especially, if the variability of number of instances in bags is high.

### Warning

Neural network created using this library has to consists only of instances of BagModel. It is not possible to combine **mil.BagModel** with other **torch.nn** modules due to correct pass of information about ids. However, preNN and afterNN neural networks can be of arbitrary form as long as the length of output of prepNN matches the length of input of afterNN.

```python
# --- WRONG ---
model = torch.nn.Sequential(
     torch.nn.Linear(10, 10, bias = True),
     mil.BagModel(prepNN, afterNN, torch.mean),
     torch.nn.Linear(10, 1, bias = True)
     )
# --- WRONG ---
```

## Examples
[musk1](https://github.com/jakubmonhart/mil_pytorch/blob/master/examples/musk/example_musk1.ipynb)

[musk2](https://github.com/jakubmonhart/mil_pytorch/blob/master/examples/musk/example_musk2.ipynb)

[musk1 with 3d data representation](https://github.com/jakubmonhart/mil_pytorch/blob/master/examples/musk/example_musk1_3d.ipynb)

[musk2 with 3d data representation](https://github.com/jakubmonhart/mil_pytorch/blob/master/examples/musk/example_musk2_3d.ipynb)

[bag\_of\_bags](https://github.com/jakubmonhart/mil_pytorch/blob/master/examples/bag_of_bags/example_bag_of_bags.ipynb)
