import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNetwork(nn.Module):
    def __init__(self, board_size: int = 5):
        super().__init__()
        self.board_size = board_size
        self.initial_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.convolutional_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.convolutional_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        flat_features = 16 * board_size * board_size
        self.value_layer = nn.Linear(in_features=flat_features, out_features=1)
        self.policy_layer = nn.Linear(in_features=flat_features, out_features=board_size*board_size)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.initial_conv(x))
        x = torch.relu(self.convolutional_1(x))
        x = torch.relu(self.convolutional_2(x))

        flat = torch.flatten(x, start_dim=1)

        policy_logits = self.policy_layer(flat)
        value = torch.tanh(self.value_layer(flat))

        return (policy_logits, value)
