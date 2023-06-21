import torch.nn as nn

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

        self.output = nn.CrossEntropyLoss()

    def forward(self,x):
        s = self.model(x)
        o = self.output(s)