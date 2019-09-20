# mil_pytorch - multiple instance learning model implemented in pytorch
This library consists mainly of BagModel and MilDataset

```python
from mil_pytorch.mil import BagModel, MilDataset
```

BagModel is subclass of torch.nn.Module (see https://pytorch.org/docs/stable/nn.html#torch.nn.Module).  
MilDataset is subclass of torch.utils.data.dataset (see https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).  

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

To use torch.utils.data.DataLoader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) you need to use custom collate function.
```python
from torch.utils.data import DataLoader
import mil_pytorch.mil as mil

dataloader = DataLoader(dataset = dataset, batch_size = batch_size, collate_fn = mil.collate_fn)
```
