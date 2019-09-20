# mil_pytorch - multiple instance learning model implemented in pytorch
This library consists mainly of BagModel and MilDataset

```python
from mil_pytorch.mil import BagModel, MilDataset
```

BagModel is subclass of torch.nn.Module (see https://pytorch.org/docs/stable/nn.html#torch.nn.Module).  
MilDataset is subclass of torch.utils.data.dataset (see https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) 
