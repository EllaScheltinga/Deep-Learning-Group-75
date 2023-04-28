from .graph_res import GraphRes

################################################################################################
# Access functions #############################################################################
################################################################################################
import torch


def by_name(name: str) -> torch.nn.Module.__class__:
    if name == "graph_res":
        return GraphRes
    else:
        raise NotImplementedError(f"Network {name} is not implemented!")