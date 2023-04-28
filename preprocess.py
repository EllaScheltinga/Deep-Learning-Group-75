import torch
from torch_geometric.transforms import Cartesian
from torch_geometric.transforms import FixedPoints
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import torch_cluster

import random as r

def normalize_time(ts: torch.Tensor, beta: float = 0.5e-5):
    return (ts - torch.min(ts)) * beta

def sub_sampling(data: Data, n_samples: int, sub_sample: bool):
  if sub_sample:
      n_samples = 3500
      sampler = FixedPoints(num=n_samples, allow_duplicates=False, replace=False)
      return sampler(data)

def pre_transform_one(data: Data, class_id):

  pre_processing_params = {"r": 5.0, "d_max": 32, "n_samples": 25000, "sampling": True}
  params = pre_processing_params

  # Cut-off window of highest increase of events.
  window_us = 50 * 1000
  t = data.pos[data.num_nodes // 2, 2]
  original_node_length = data.num_nodes

  index1 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t) - 1, 0, data.num_nodes - 1)
  index0 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t-window_us) - 1, 0, data.num_nodes - 1)
  for key, item in data:
      if torch.is_tensor(item) and item.size(0) == original_node_length and item.size(0) != 1:
          data[key] = item[index0:index1, :]
          
  # Coarsen graph by uniformly sampling n points from the event point cloud.
  data = sub_sampling(data, n_samples=params["n_samples"], sub_sample=params["sampling"])
  # Re-weight temporal vs. spatial dimensions to account for different resolutions.
  data.pos[:, 2] = normalize_time(data.pos[:, 2])
  # Radius graph generation.
  data.edge_index = radius_graph(data.pos, r=params["r"], max_num_neighbors=params["d_max"])
  data.y = torch.tensor([class_id])
  edge_attr = Cartesian(norm=True, cat=False, max_value=5.0)
  data = edge_attr(data)

  return data

def pre_transform_all(data, class_id, device):
  pre_transformed_data = []
  for item in data:
    pre_transformed_data.append(pre_transform_one(item, class_id).to(device))
  return pre_transformed_data