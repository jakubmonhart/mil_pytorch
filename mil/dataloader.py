
import torch 
from torch.utils.data import Dataset, DataLoader


class MilDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.unique_bag_ids = data.bagID.unique()

    def __len__(self):
        return self.data.bagID.nunique()
    
    def __getitem__(self, index):
        bag_id = self.unique_bag_ids[index]
        index_data = self.data[self.data.bagID==bag_id]
        index_features =  torch.tensor(index_data[index_data.columns[3:]].values)
        index_label = torch.tensor(index_data.response.unique())

        return index_features, index_label



def get_train_test_dataloader(train_data, test_data, train_batch_size=1, test_batch_size=1):
    if (train_batch_size !=1) | (test_batch_size != 1):
        raise NotImplementedError
    
    train_data = MilDataset(train_data)
    test_data = MilDataset(test_data)

    train_dl = DataLoader(dataset=train_data, batch_size=train_batch_size) # Using custom collate_fn mil.collate
    test_dl = DataLoader(dataset=test_data, batch_size=test_batch_size)

    return train_dl, test_dl