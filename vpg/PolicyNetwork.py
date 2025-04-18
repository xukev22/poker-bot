import torch.nn as nn, torch.nn.functional as F


class VPGPolicy(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.logits = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits(x)  # apply softmax + mask at sampling time
