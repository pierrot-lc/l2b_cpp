"""Attention over childs embeddings.
"""
import torch
import torch.nn as nn


from src.models.mlp import MLP


class AttentionChilds(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_layers: int,
        dim_embedding: int,
        dropout: float,
        nhead: int,
        dim_feedforward_transformer: int,
    ):
        super().__init__()
        self.input_size = input_size

        self.project_layer = nn.Sequential(
            nn.Linear(input_size, dim_embedding),
            nn.LayerNorm(dim_embedding),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = dim_embedding,
            nhead = nhead,
            dim_feedforward = dim_feedforward_transformer,
            dropout = dropout,
            batch_first = True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out = nn.Sequential(
            nn.Linear(dim_embedding, 1)
        )

    def forward(self,
            x: torch.FloatTensor,
            key_mask: torch.BoolTensor = None,
        ) -> torch.FloatTensor:
        """
        Args
        ----
            x: Batch of childs.
                Shape of [batch_size, n_childs, input_size].
            key_mask: Mask for the attention module.
                Shape of [batch_size, n_childs].

        Return
        ------
            y: Batch of predicted values.
                Shape of [batch_size, n_childs].
        """
        emb = self.project_layer(x)
        emb = self.encoder(emb, src_key_padding_mask=key_mask)
        y = self.out(emb)  # [batch_size, n_childs, 1]
        return y.squeeze(dim=-1)  # [batch_size, n_childs]

    def from_config(config: dict):
        """Load the model from the saved config file in WandB.
        """
        config['input_size'] = int(config['input_size']['value'])
        config['n_layers'] = int(config['n_layers']['value'])
        config['dim_embedding'] = int(config['dim_embedding']['value'])
        config['dropout'] = float(config['dropout']['value'])
        config['nhead'] = int(config['nhead']['value'])
        config['dim_feedforward_transformer'] = int(config['dim_feedforward_transformer']['value'])

        return AttentionChilds(
            config['input_size'],
            config['n_layers'],
            config['dim_embedding'],
            config['dropout'],
            config['nhead'],
            config['dim_feedforward_transformer'],
        )
