"""Basic MLP.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, n_layers, n_hidden, dropout):
        super(MLP, self).__init__()
        self.input_size = input_size

        self.project_layer = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.LayerNorm(n_hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )
            for _ in range(n_layers)
        ])

        self.output_layer = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.project_layer(x)
        for layer in self.res_layers:
            x = x + layer(x)
        return self.output_layer(x)
