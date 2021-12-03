from torch import nn
import torch.utils.data.dataloader


class Net(nn.Module):
    def __init__(self, n_actions):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)

        return self.fc(self.conv(x).view(x.shape[0], -1))

