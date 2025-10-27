import torch
from torch import nn


class LowerTriangularBCELoss(nn.Module):
    def __init__(self):
        """
        Compute BCE loss only for the lower-triangular part of the similarity matrix
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, mat, target_mat):
        B, L = mat.shape[:2]

        mask = torch.tril(torch.ones(L, L, device=mat.device), diagonal=-1).bool()
        mask = mask.unsqueeze(0).expand(B, -1, -1)

        # loss_mat contains many -inf
        loss_mat = self.bce(mat, target_mat)

        # There are more zeros than ones
        loss_mat = torch.clamp(loss_mat, max=10.0)  # Prevent extreme values

        # eliminate nan values from mask
        mask = mask & (~torch.isnan(loss_mat))

        return loss_mat[mask].mean()

        # return loss_mat.masked_fill(~mask, 0.0).sum() / mask.sum()

class EdgeMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

        self.weight = torch.tensor([3.0] + [1.0]*6 + [3.0]).view(1, 8, 1)
        self.weight = self.weight / self.weight.sum()

    def forward(self, pred_edges, target_edges):
        """
        pred_edges: (B, ..., 8, 2)
        target_edges: (B, ..., 8, 2)
        """
        pred_edges = pred_edges.view(-1, 8, 2)
        target_edges = target_edges.view(-1, 8, 2)

        loss = self.mse(pred_edges, target_edges)

        # Give higher weight to the endpoints
        weights = self.weight.to(pred_edges.device)

        return (loss * weights).mean()

# Freaking useless, does not converge at all
# class ChamferLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, pred_edges, target_edges):
#         """
#         pred_edges: (B, ..., 8, 2)
#         target_edges: (B, ..., 8, 2)
#         """
#         B = pred_edges.shape[0]
#         pred_edges = pred_edges.view(B, -1, 16)   # (B, L, 8, 2)
#         target_edges = target_edges.view(B, -1, 16)
#
#         cost_matrix = torch.cdist(pred_edges, target_edges, p=2)  # (B, L, L)
#
#         loss_pred_to_target, _ = cost_matrix.min(dim=2)  # (B, L)
#         loss_target_to_pred, _ = cost_matrix.min(dim=1)  # (B, L)
#
#         return (loss_pred_to_target.mean() + loss_target_to_pred.mean()) / 2.0



class CountLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.threshold = threshold

    def forward(self, affinity_mat, target_mat):
        """
        pred_count: (B, L)
        target_count: (B, L)
        """
        B, L = affinity_mat.shape[:2]

        target_count = (target_mat > 0.5).float().sum(dim=2)  # (B, L)
        pred_count_soft = torch.sigmoid(affinity_mat).sum(dim=2)  # (B, L)
        # pred_count_hard = (pred_count_soft > self.threshold).float()
        loss = self.mse(pred_count_soft, target_count)
        return loss


def getTargetUniquenessMask(duplicate_edges: torch.Tensor, target_edges: torch.Tensor):
    # duplicate_edges: (B, N_edges, N_interp, 2)
    # target_edges: (B, N_edges, N_interp, 2)
    B, L = duplicate_edges.shape[:2]

    target_edges = target_edges.to(duplicate_edges.dtype)
    rev_target_edges = target_edges.flip(dims=[2]).flatten(2)
    target_edges = target_edges.flatten(2)  # (B, M, 2)
    all_target_edges = torch.cat([target_edges, rev_target_edges], dim=1)  # (B, 2M, 2)

    cost_matrices = torch.cdist(duplicate_edges.detach().flatten(2), all_target_edges, p=2)    # (B, L, 2M)

    # Step 2. rearrange the target_seq with the matching, so input seq and target_seq have 1-1 correspondence
    # if target_seq[j] is the nearest to input_seq[i], then matched_target_seq[i] = target_seq[j]
    nearest_match = cost_matrices.argmin(dim=2)  # (B, 2M)
    batch_idx = torch.arange(B, device=duplicate_edges.device).unsqueeze(1).expand(-1, L)  # (B, L)
    matched_target_seq = all_target_edges[batch_idx, nearest_match]  # (B, L, D)

    dist_mat = torch.cdist(matched_target_seq, matched_target_seq)
    sim_mat = (dist_mat < 0.01).float()

    # STEP 2. keep only the similarity scores of previous elements for each element
    mask = torch.tril(torch.ones(L, L, device=sim_mat.device, dtype=sim_mat.dtype), diagonal=-1).bool()
    mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)
    sim_mat = sim_mat.masked_fill(~mask, 0.0)

    # STEP 3. get the maximum previous similarity score
    # high score means exists previous similar element
    max_previous_sim, _ = sim_mat.max(dim=2)  # (B, L)

    # STEP 4. construct gating, first appear element maps to 1, others map tp 0
    gate = (1 - max_previous_sim) #(B, L, 1)
    return sim_mat, (gate > 0.5).to(duplicate_edges.dtype)
