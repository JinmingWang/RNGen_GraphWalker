import torch
from typing import List, Tuple, Set, FrozenSet, Literal, Dict
from jaxtyping import Float, Bool
from einops import rearrange, reduce
import random

Tensor = torch.Tensor
Segment = Float[Tensor, "N_interp 2"]
Route = Float[Tensor, "L_route N_interp 2"]
Trajectory = Float[Tensor, "L_traj 2"]
Graph = Float[Tensor, "N_segs N_interp 2"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




