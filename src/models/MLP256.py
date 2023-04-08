import torch


class MLP256(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(512, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(512, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(512, 256),
        )

    def forward(self, x):
        y = self.mlp(x)
        return y
