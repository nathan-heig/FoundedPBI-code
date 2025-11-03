import ast
import torch
from torch import nn
import torch.nn.functional as F
from pbi_utils.logging import Logging
from pbi_models.classifiers.abstract_classifier import AbstractNNClassifier

logger = Logging()

class ConvBlock1D(nn.Module):
    """Simple Conv1D + ReLU + MaxPool1D block."""
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return x
    
class BranchCNN(nn.Module):
    """
    Branch CNN for either bacteria or phage input.

    Coauthored by ChatGPT
    """
    def __init__(self, 
                 conv_params: list[tuple[int, int, int]]):
        """
        conv_params: list of (out_channels, kernel_size, pool_size)
        Example: [(64, 3, 5), (32, 10, 5), (32, 10, 5)]
        """
        super().__init__()
        layers = []
        in_ch = 1
        for (out_ch, k, p) in conv_params:
            layers.append(ConvBlock1D(in_ch, out_ch, k, p))
            in_ch = out_ch
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CNNClassifier(AbstractNNClassifier):
    """
    General CNN classifier with two branches (one for bacteria, one for phages).
    Coauthored by ChatGPT
    """
    def __init__(self, bacterium_embed_dim: int, phage_embed_dim: int, bacterium_conv_params: list[tuple[int, int, int]] | str, phage_conv_params: list[tuple[int, int, int]] | str,
                 dense_dim: int | str = 128, dense_dropout: float | str = 0.5):
        
        super().__init__(bacterium_embed_dim, phage_embed_dim)

        # Sanity checks
        self._sanity_checks(bacterium_embed_dim, phage_embed_dim, bacterium_conv_params, phage_conv_params, dense_dim, dense_dropout)

        # Convert params to dict
        if isinstance(bacterium_conv_params, str):
            bacterium_conv_params = self._parse_branch_params(bacterium_conv_params)
        if isinstance(phage_conv_params, str):
            phage_conv_params = self._parse_branch_params(phage_conv_params)

        self.bacterium_conv_params = bacterium_conv_params
        self.phage_conv_params = phage_conv_params

        # Branches
        self.bacteria_branch = BranchCNN(bacterium_conv_params)
        self.phage_branch = BranchCNN(phage_conv_params)

        # Determine flatten sizes dynamically
        self.bacteria_feat_dim = self._get_flatten_size(bacterium_embed_dim, self.bacteria_branch)
        self.phage_feat_dim = self._get_flatten_size(phage_embed_dim, self.phage_branch)
        concat_dim = self.bacteria_feat_dim + self.phage_feat_dim

        # Dense layers
        self.fc1 = nn.Linear(concat_dim, int(dense_dim))
        self.dropout = nn.Dropout(float(dense_dropout))
        self.fc2 = nn.Linear(int(dense_dim), 2)

    def _sanity_checks(self, bacterium_embed_dim, phage_embed_dim, bacterium_conv_params, phage_conv_params, dense_dim, dense_dropout):
        if isinstance(bacterium_conv_params, list):
            for tpl in bacterium_conv_params:
                assert isinstance(tpl, tuple) and len(tpl) == 3 and all(isinstance(x, int) for x in tpl), f"bacterium_conv_params must be a list of tuples of three integers or a string. Got: {bacterium_conv_params}"
        else:
            assert isinstance(bacterium_conv_params, str), f"bacterium_conv_params must be either a list of tuples of three integers or a string. Got: {bacterium_conv_params}"
        if isinstance(phage_conv_params, list):
            for tpl in phage_conv_params:
                assert isinstance(tpl, tuple) and len(tpl) == 3 and all(isinstance(x, int) for x in tpl), f"phage_conv_params must be a list of tuples of three integers or a string. Got: {phage_conv_params}"
        else:
            assert isinstance(phage_conv_params, str), f"phage_conv_params must be either a list of tuples of three integers or a string. Got: {phage_conv_params}"
        assert isinstance(dense_dim, (int, str)), f"dense_dim must be either an integer or a string. Got: {type(dense_dim)}: {dense_dim}"
        assert isinstance(dense_dropout, (float, str)), f"dense_dropout must be either a float or a string. Got: {type(dense_dropout)}: {dense_dropout}"
        assert isinstance(bacterium_embed_dim, int), f"bacterium_embed_dim must be an integer. Got {type(bacterium_embed_dim)}: {bacterium_embed_dim}"
        assert isinstance(phage_embed_dim, int), f"phage_embed_dim must be an integer. Got {type(phage_embed_dim)}: {phage_embed_dim}"

    def _parse_branch_params(self, params_str: str) -> list[tuple[int, int, int]]:
        """Parse branch parameters from string representation."""
        # Using ast.literal_eval for safe evaluation
        return ast.literal_eval(params_str)

    def _get_flatten_size(self, input_len, branch):
        """Compute flattened size automatically for given input length."""
        x = torch.zeros(1, 1, input_len)
        with torch.no_grad():
            x = branch(x)
        return x.numel()

    def forward(self, bacterium_emb: torch.Tensor, phage_emb: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            bacterium_emb: [batch, emb_dim]
            phage_emb:     [batch, emb_dim]
        Returns:
            logits: [batch, num_classes]
        """
        # Reshape for Conv1D: [batch, channels=1, seq_len]
        bacterium_emb = bacterium_emb.unsqueeze(1)
        phage_emb = phage_emb.unsqueeze(1)

        # Branch processing
        x_b = self.bacteria_branch(bacterium_emb)
        x_p = self.phage_branch(phage_emb)

        # Flatten & concatenate
        x_b = torch.flatten(x_b, start_dim=1)
        x_p = torch.flatten(x_p, start_dim=1)
        x = torch.cat((x_b, x_p), dim=1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # No softmax, return raw logits
        return x

    
    def name(self) -> str:
        return self.__repr__()
        
    def __repr__(self) -> str:
        return f"""CNNClassifier(bacterium_conv_params={self.bacterium_conv_params}, phage_conv_params={self.phage_conv_params}, dense_dim={self.fc1.out_features}, dense_dropout={self.dropout.p})"""
    
class BasicCNNClassifier(CNNClassifier):
    def __init__(self, bacterium_embed_dim: int, phage_embed_dim: int, cnn_params: list[tuple[int, int, int]] | str, dense_dim: int | str = 128, dense_dropout: float | str = 0.5):
        super().__init__(bacterium_embed_dim, phage_embed_dim, [], [], dense_dim, dense_dropout)

        if isinstance(cnn_params, str):
            self.params = self._parse_branch_params(cnn_params)
        else:
            self.params = cnn_params

        self.cnn = BranchCNN(self.params)

        self.flat_size = self._get_flatten_size(bacterium_embed_dim + phage_embed_dim, self.cnn)

        # Dense layers
        self.fc1 = nn.Linear(self.flat_size, int(dense_dim))
        self.dropout = nn.Dropout(float(dense_dropout))
        self.fc2 = nn.Linear(int(dense_dim), 2)

        
    def forward(self, bacterium_emb: torch.Tensor, phage_emb: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            bacterium_emb: [batch, emb_dim]
            phage_emb:     [batch, emb_dim]
        Returns:
            logits: [batch, num_classes]
        """
        # Reshape for Conv1D: [batch, channels=1, seq_len]
        bacterium_emb = bacterium_emb.unsqueeze(1)
        phage_emb = phage_emb.unsqueeze(1)

        x = torch.cat((bacterium_emb, phage_emb), dim=2)

        x = self.cnn(x)

        x = torch.flatten(x, start_dim=1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # No softmax, return raw logits
        return x