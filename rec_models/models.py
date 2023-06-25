import torch.nn as nn
import torch.nn.functional as F
class RecNet(nn.Module):
    def __init__(self,n_features) -> None:
        super(RecNet,self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(n_features,32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(8,2)
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self,x,y=None):
        logits = self.model(x)
        outputs = (logits,)
        if y:
            loss_value = self.loss(logits,y)
            outputs = (loss_value,)+outputs
        return outputs

class WideComponent(nn.Module):
    def __init__(self,n_features):
        super(WideComponent,self).__init__()
        self.linear = nn.Linear(n_features=n_features,out_features=1)
    def forward(self,x):
        return self.linear(x)
    
class DnnComponent(nn.Module):
    def __init__(self,hidden_units,dropout=0.):
        super(DnnComponent,self).__init__()

        self.dnn = nn.ModuleList([nn.Linear(layer[0],layer[1])for layer in list(zip(hidden_units[:-1],hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,x):

        for linear in self.dnn:
            x = linear(x)
            x = F.relu(x)
        
        x = self.dropout(x)

        return x