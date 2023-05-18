import torch.nn as nn

from transformers import BertModel


class DeepBert(nn.Module):
    def __ini__(self, name_or_path, freeze_bert: bool = True):
        super(DeepBert, self).__init__()
        self.bert = BertModel.from_pretrained(name_or_path)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        outputs = self.bert(**inputs)


class MLPNet(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        self.output = nn.Sigmoid()

    def forward(self, x1, x2):
        o1 = self.model(x1)
        o2 = self.model(x2)
        return self.output(o1-o2)

    def predict(self, x):
        return self.model(x)
