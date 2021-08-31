import torch.nn as nn
import torch.nn.functional as F
import torch

class LinearModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.lin1 = nn.Linear(self.num_features, 150)
        self.lin2 = nn.Linear(150, 150)
        self.lin3 = nn.Linear(150, 1)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(0.25)


    def forward(self, xin):
        x = F.relu(self.lin1(xin))

        for y in range(8):
            x = F.relu(self.lin2(x))

        x = self.dropout(x)

        x = self.sigmoid(self.lin3(x))
        return x
