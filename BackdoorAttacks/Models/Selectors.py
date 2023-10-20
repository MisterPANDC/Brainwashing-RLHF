import torch
from torch import nn

class selector1(nn.Module):
    def __init__(self, emb_dimension=100):
        super().__init__()
        self.emb_dimension = emb_dimension
        self.fc1 = nn.Linear(self.emb_dimension, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
        return x

class selector2(nn.Module):
    def __init__(self, emb_dimension=100):
        super().__init__()
        self.emb_dimension = emb_dimension
        self.fc1 = nn.Linear(self.emb_dimension, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 256)
        self.fc5 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(64, 16)
        self.fc7 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        return x

class selector4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x