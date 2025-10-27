# components.py
from JimmyTorch.Models import *


class Deduplicator(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_hidden: int,
                 threshold: float):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.threshold = threshold

        self.x_proj = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.SiLU(inplace=True),
            nn.Linear(d_in, d_in),
            nn.SiLU(inplace=True),
            nn.Linear(d_in, d_hidden),
            # Rearrange("B L (H D) -> B L (H D)", H=n_heads, D=d_head)
        )   # ((Batch, Heads), N_segs, D_seg)

        self.sim_mat_proj = nn.Sequential(
            # Multi-head squeeze
            # Permute(0, 2, 3, 1),   # (B, L, L, H)
            nn.Linear(4 * d_hidden, d_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(d_hidden, 1),
            nn.Flatten(2),
            nn.Tanh(),  # (B, L, L)
        )

    def forward(self, duplicate_edges, x):
        # duplicate_edges: (B, L, D)
        B, L, D = duplicate_edges.shape

        # STEP 1. get pair-wise similarity matrix
        # x: (B*H, N_segs, D_seg)
        x = self.x_proj(x)

        xi = x.unsqueeze(2).expand(B, L, L, self.d_hidden)  # (B, L, L, HD)
        xj = x.unsqueeze(1).expand(B, L, L, self.d_hidden)
        features = torch.cat([xi, xj, xi - xj, xi * xj], dim=-1)    # (B, L, L, 4*HD)

        # affinity_mat: -inf to inf
        # we output logits instead of probabilities, but there can be fp16 overflow issue, gradient explosion issue, etc.
        # So here we use tanh(x) * 15.0 to limit the output range to [-15, 15],
        # which is equivalent to [3e-7, 1-3e-7] in probability space
        affinity_mat = self.sim_mat_proj(features) * 15.0

        # STEP 2. keep only the similarity scores of previous elements for each element
        mask = torch.tril(torch.ones(L, L, device=duplicate_edges.device), diagonal=-1).bool()
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)
        affinity_mat = affinity_mat.masked_fill(~mask, -torch.inf)
        return affinity_mat

    def getGraphs(self, duplicate_edges, affinity_mat):
        max_previous_score, _ = affinity_mat.max(dim=2)
        first_appear_likelihood = 1 - torch.sigmoid(max_previous_score)
        select_mask = first_appear_likelihood > self.threshold
        batch_edges = torch.unbind(duplicate_edges, dim=0)
        graphs = [edges[select_mask[b]] for b, edges in enumerate(batch_edges)]
        return graphs


class Res1D(nn.Module):
    def __init__(self,
                 d_in: int,
                 expansion: int = 2,
                 ):
        super().__init__()

        self.res_branch = nn.Sequential(
            Conv1DInSiLU(d_in, d_in * expansion, 3, 1, 1),
            Conv1DInSiLU(d_in * expansion, d_in * expansion, 3, 1, 1),
            nn.Conv1d(d_in * expansion, d_in, 3, 1, 1),
        )
        nn.init.zeros_(self.res_branch[-1].weight)
        nn.init.zeros_(self.res_branch[-1].bias)

    def forward(self, x):
        return self.res_branch(x) + x


class VAEBlock(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_forward: int,
                 n_heads: int,
                 n_walks: int,
                 dropout: float = 0.0):
        super().__init__()

        self.d_in = d_in
        self.d_forward = d_forward
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_walks = n_walks

        self.ln = nn.LayerNorm(d_in)
        self.attn = MHSA(d_in, n_heads, dropout)

        # Typical ff layer processes each token (edge) independently,
        # but each edge has relationships with its neighbor edges in the same walk,
        # so we use conv1d to replace the ff layer to capture local patterns instead of private tokens
        self.ff = nn.Sequential(
            Rearrange("B (N_walks L_walk) D -> (B N_walks) D L_walk", N_walks=n_walks),
            InSiLUConv1D(d_in, d_forward, 3, 1, 1),
            InSiLUConv1D(d_forward, d_in, 3, 1, 1),
            Rearrange("(B N_walks) D L_walk -> B (N_walks L_walk) D", N_walks=n_walks),
        )
        nn.init.zeros_(self.ff[2][2].weight)
        nn.init.zeros_(self.ff[2][2].bias)

    def forward(self, x):
        # x: (B, N, L, D)
        x = x + self.attn(self.ln(x))
        return x + self.ff(x)


class DiTBlock(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_time: int,
                 d_forward: int,
                 n_heads: int,
                 n_walks: int,
                 dropout: float = 0.0):
        super().__init__()

        self.d_in = d_in
        self.d_time = d_time
        self.d_forward = d_forward
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_walks = n_walks

        self.time_proj = nn.Sequential(
            nn.Linear(d_time, d_time),
            nn.SiLU(inplace=True),
            nn.Linear(d_time, d_in * 4)
        )
        # initially, the scale is 1 and the shift is 0
        nn.init.zeros_(self.time_proj[-1].weight)
        nn.init.zeros_(self.time_proj[-1].bias)

        self.ln1 = nn.LayerNorm(d_in)
        self.attn = MHSA(d_in, n_heads, dropout)

        # Typical ff layer processes each token (edge) independently,
        # but each edge has relationships with its neighbor edges in the same walk,
        # so we use conv1d to replace the ff layer to capture local patterns instead of private tokens
        self.ln2 = nn.LayerNorm(d_in)
        self.ff = nn.Sequential(
            Rearrange("B (N_walks L_walk) D -> (B N_walks) D L_walk", N_walks=n_walks),
            nn.SiLU(inplace=True),
            nn.Conv1d(d_in, d_forward, 3, 1, 1),
            InSiLUConv1D(d_forward, d_in, 3, 1, 1),
            Rearrange("(B N_walks) D L_walk -> B (N_walks L_walk) D", N_walks=n_walks),
        )
        nn.init.zeros_(self.ff[-2][2].weight)
        nn.init.zeros_(self.ff[-2][2].bias)

        self.out_proj = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Linear(d_in, d_in)
        )

    def forward(self, x, t):
        # x: (B, N, L, D)
        identity = x
        scale1, shift1, scale2, shift2 = self.time_proj(t).chunk(4, dim=-1)     # (B, N, 4*D) -> 4 * (B, N, D)
        x = x + self.attn(self.ln1(x) * (1 + scale1) + shift1)
        x = x + self.ff(self.ln2(x) * (1 + scale2) + shift2)
        return self.out_proj(x) + identity


class InterWalksVAEBlock(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_forward: int,
                 n_heads: int,
                 n_walks: int,
                 dropout: float = 0.0):
        super().__init__()

        self.d_in = d_in
        self.d_forward = d_forward
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_walks = n_walks

        self.ln = nn.LayerNorm(d_in)
        self.attn = MHSA(d_in, n_heads, dropout)

        # Typical ff layer processes each token (edge) independently,
        # but each edge has relationships with its neighbor edges in the same walk,
        # so we use conv1d to replace the ff layer to capture local patterns instead of private tokens
        self.ff = nn.Sequential(
            nn.LayerNorm(d_in),
            MLP([d_in, d_forward, d_in], nn.SiLU(inplace=True))
        )
        nn.init.zeros_(self.ff[1][-1].weight)
        nn.init.zeros_(self.ff[1][-1].bias)

    def forward(self, x):
        # x: (B, N, L, D)
        x = x + self.attn(self.ln(x))
        return x + self.ff(x)


class InterWalksDiTBlock(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_time: int,
                 d_forward: int,
                 n_heads: int,
                 n_walks: int,
                 dropout: float = 0.0):
        super().__init__()

        self.d_in = d_in
        self.d_time = d_time
        self.d_forward = d_forward
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_walks = n_walks

        self.time_proj = nn.Sequential(
            nn.Linear(d_time, d_time),
            nn.SiLU(inplace=True),
            nn.Linear(d_time, d_in * 4)
        )
        # initially, the scale is 1 and the shift is 0
        nn.init.zeros_(self.time_proj[-1].weight)
        nn.init.zeros_(self.time_proj[-1].bias)

        self.ln1 = nn.LayerNorm(d_in)
        self.attn = MHSA(d_in, n_heads, dropout)

        # Typical ff layer processes each token (edge) independently,
        # but each edge has relationships with its neighbor edges in the same walk,
        # so we use conv1d to replace the ff layer to capture local patterns instead of private tokens
        self.ln2 = nn.LayerNorm(d_in)
        self.ff = MLP([d_in, d_forward, d_in], nn.SiLU(inplace=True))
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

        self.out_proj = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Linear(d_in, d_in)
        )

    def forward(self, x, t):
        # x: (B, N, L, D)
        identity = x
        scale1, shift1, scale2, shift2 = self.time_proj(t).chunk(4, dim=-1)     # (B, N, 4*D) -> 4 * (B, N, D)
        x = x + self.attn(self.ln1(x) * (1 + scale1) + shift1)
        x = x + self.ff(self.ln2(x) * (1 + scale2) + shift2)
        return self.out_proj(x) + identity


def matchJoints(edges: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    # edges: (B, n_edges, 5), joints: (B, n_edges, n_edges)
    joints = (joints >= 0.5).to(torch.float32)
    # remove diagonal
    joints = joints - torch.eye(joints.shape[0], device=joints.device)

    segs = edges[:, :4]  # Only use (x1, y1, x2, y2)
    N = segs.shape[0]

    for seg_i in range(N):
        neighbor_ids = torch.nonzero(joints[seg_i]).flatten()  # Get neighbors for segment i
        if len(neighbor_ids) == 0:
            continue  # No neighbors, move to the next segment

        # Compute pairwise distances between p1 and p2 of seg_i and the neighbors
        dist_p1n1 = torch.norm(segs[seg_i, 0:2] - segs[neighbor_ids, 0:2], dim=1)  # Distance between p1 of seg_i and p1 of neighbors
        dist_p1n2 = torch.norm(segs[seg_i, 0:2] - segs[neighbor_ids, 2:4], dim=1)  # Distance between p1 of seg_i and p2 of neighbors
        dist_p2n1 = torch.norm(segs[seg_i, 2:4] - segs[neighbor_ids, 0:2], dim=1)  # Distance between p2 of seg_i and p1 of neighbors
        dist_p2n2 = torch.norm(segs[seg_i, 2:4] - segs[neighbor_ids, 2:4], dim=1)  # Distance between p2 of seg_i and p2 of neighbors

        # Stack the distances and find the minimum for each neighbor
        distances = torch.stack([dist_p1n1, dist_p1n2, dist_p2n1, dist_p2n2], dim=0)  # Shape (4, N_i)
        min_dist, min_idx = torch.min(distances, dim=0)  # Get minimum distance index for each neighbor

        # Identify matching points
        p1n1_match = min_idx == 0
        p1n2_match = min_idx == 1
        p2n1_match = min_idx == 2
        p2n2_match = min_idx == 3

        # Compute the mean of the matched points
        p1_sum = segs[seg_i, 0:2] + segs[neighbor_ids][p1n1_match, 0:2].sum(dim=0) + segs[neighbor_ids][p1n2_match][:, 2:4].sum(dim=0)
        p1_mean = p1_sum / (1 + p1n1_match.sum() + p1n2_match.sum())  # Take average

        p2_sum = segs[seg_i, 2:4] + segs[neighbor_ids][p2n1_match, 0:2].sum(dim=0) + segs[neighbor_ids][p2n2_match][:, 2:4].sum(dim=0)
        p2_mean = p2_sum / (1 + p2n1_match.sum() + p2n2_match.sum())  # Take average

        # Update current segment's points
        segs[seg_i, 0:2] = p1_mean
        segs[seg_i, 2:4] = p2_mean

        # Update the matched neighbors' points accordingly
        segs[neighbor_ids[p1n1_match], 0:2] = p1_mean  # p1n1: update p2 of neighbors
        segs[neighbor_ids[p1n2_match], 2:4] = p1_mean  # p1n2: update p1 of neighbors
        segs[neighbor_ids[p2n1_match], 0:2] = p2_mean  # p2n1: update p2 of neighbors
        segs[neighbor_ids[p2n2_match], 2:4] = p2_mean  # p2n2: update p1 of neighbors

        # Mark processed neighbors to avoid double-processing
        joints[neighbor_ids, seg_i] = 0

    return torch.cat([segs, edges[:, 4:]], dim=-1)
