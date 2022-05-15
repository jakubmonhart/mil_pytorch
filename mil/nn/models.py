import torch 
import torch.nn as nn
import torch.nn.functional as F

class mi_Net(nn.Module):
    def __init__(self, input_dim=166, pooling_method="mean"):
        super().__init__()

        self.fc1 = nn.Linear(in_features=input_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.fc4 = nn.Linear(in_features=64, out_features=1)
        if pooling_method == "mean":
            self.pooling_method = torch.mean
        elif pooling_method == "max":
            self.pooling_method = torch.max
        else:
            raise NotImplementedError
    
    def forward(self, input):    
        
        x = input.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x) 
        x = torch.sigmoid(self.fc4(x))       
        return self.pooling_method(x, dim=1, keepdim=False)


class MI_Net(nn.Module):
    def __init__(self, input_dim=166, pooling_method="mean", dropout=True):
        super().__init__()

        self.fc1 = nn.Linear(in_features=input_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.dropout = dropout
        if dropout:
            self.dropout_1 = nn.Dropout(p=0.5, inplace=False)
            self.dropout_2 = nn.Dropout(p=0.5, inplace=False)
            self.dropout_3 = nn.Dropout(p=0.5, inplace=False)
        
        if pooling_method == "mean":
            self.pooling_method = torch.mean
        elif pooling_method == "max":
            self.pooling_method = torch.max
        else:
            raise NotImplementedError
        
        self.fc4 = nn.Linear(in_features=64, out_features=1)
    
    def forward(self, input):    
        
        x = input.float()
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout_1(x)
        
        x = F.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout_2(x)
        
        x = F.relu(self.fc3(x))
        if self.dropout:
            x = self.dropout_3(x)
        x = self.pooling_method(x, dim=1, keepdim=False)       
        x = self.fc4(x)

        return torch.sigmoid(x)


class MI_Net_RC(nn.Module):
    def __init__(self, input_dim=166, dropout=False, pooling_method="mean"):
        super().__init__()

        self.fc1 = nn.Linear(in_features=input_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=1)
        self.dropout = dropout
        if dropout:
            self.dropout_1 = nn.Dropout(p=0.5, inplace=False)
            self.dropout_2 = nn.Dropout(p=0.5, inplace=False)
            self.dropout_3 = nn.Dropout(p=0.5, inplace=False)
        
        if pooling_method == "mean":
            self.pooling_method = torch.mean
        elif pooling_method == "max":
            self.pooling_method = torch.max
        else:
            raise NotImplementedError
        
    def forward(self, input):    
        
        x = input.float()
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout_1(x)
        r1 = self.pooling_method(x, dim=1, keepdim=False)
        x = F.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout_2(x)
        r2 = self.pooling_method(x, dim=1, keepdim=False)
        x = F.relu(self.fc3(x))
        if self.dropout:
            x = self.dropout_3(x)
        r3 = self.pooling_method(x, dim=1, keepdim=False)        
        x = r1 + r2 + r3
        x = self.fc4(x)

        return torch.sigmoid(x)