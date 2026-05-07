"""NuGraph3 energy decoder"""
from typing import Any
import torch
from torch import nn
from torch_geometric.data import Batch
from pytorch_lightning.loggers import TensorBoardLogger
#from ....util import LogCoshLoss
from ..types import Data

class EnergyDecoder(nn.Module):
    """
    NuGraph3 energy decoder module

    Convolve interaction node embedding down to a single energy evaluation.
    Based on code from vertex head

    Args:
        interaction_features: Number of interaction node features
    """
    def __init__(self, interaction_features: int):
        super().__init__()

        # loss function
        self.loss = nn.HuberLoss(delta=1.0)

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(5.))

        # network
        self.net = nn.Linear(interaction_features, 1)

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 energy decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # run network and calculate loss
        x = self.net(data["evt"].x)
        y = data["evt"].y_en
        w = (-1 * self.temp).exp()
        loss = w * self.loss(x, y.unsqueeze(1)) + self.temp #TO CHANGE FOR 1D

        # add inference output to graph object
        data["evt"].energy = x
        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["evt"]["energy"] = data["evt"].ptr
            inc = torch.zeros(data.num_graphs, device=data["evt"].x.device)
            data._inc_dict["evt"]["energy"] = inc

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"energy/loss-{stage}"] = loss
            delta_e = (x-y).abs().mean()
            metrics[f"energy/e-resolution-{stage}"] = delta_e
        if stage == "train":
            metrics["temperature/energy"] = self.temp

        return loss, metrics

    def on_epoch_end(self,
                     logger: TensorBoardLogger,
                     stage: str,
                     epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
