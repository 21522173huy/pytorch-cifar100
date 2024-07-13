
import torch
from torch import nn
class MLP_Classifier(nn.Module):
    def __init__(self, in_features, out_features, hidden_size = 1024, activation=nn.ReLU(), p = 0.1):
        super(MLP_Classifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.activation = activation
        
        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            self.activation,
            nn.Dropout(p), 
            nn.Linear(self.hidden_size, self.out_features)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
