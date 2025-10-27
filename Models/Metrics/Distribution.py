from scipy.stats import wasserstein_distance
import torch

Tensor = torch.Tensor

def edgeLengthMetric(batch_pred_edges, batch_target_edges, reduction='mean'):
    """
    Compute the difference in the length distribution of the predicted and target segments
    :param batch_pred_edges: the predicted segments
    :param batch_target_edges: the target segments
    :param reduction: 'none' | 'mean' | 'sum'
    :return: the difference in the length distribution of the predicted and target segments
    """
    B = len(batch_pred_edges)

    # Each batch item is of shape (P, N_interp, 2) or (Q, N_interp, 2)
    # It contains P or Q segments, each segment has N_interp points in 2D

    distribution_diffs = []
    for b in range(B):
        pred_edges = batch_pred_edges[b]  # (P, N_interp, 2)
        target_edges = batch_target_edges[b]  # (Q, N_interp, 2)

        # Compute the length of each segment

        # (P, N_interp, 2) -> (P, )
        pred_lengths = torch.linalg.norm(pred_edges[:, 1:] - pred_edges[:, :-1], dim=-1).sum(dim=-1)
        # (Q, N_interp, 2) -> (Q, )
        target_lengths = torch.linalg.norm(target_edges[:, 1:] - target_edges[:, :-1], dim=-1).sum(dim=-1)
        if len(target_lengths) == 0:
            target_lengths = torch.zeros_like(pred_lengths)

        # Compute the difference in the length distribution using Wasserstein distance
        distribution_diff = wasserstein_distance(pred_lengths.cpu().numpy(), target_lengths.cpu().numpy())
        distribution_diffs.append(distribution_diff)

    if reduction == 'none':
        return torch.tensor(distribution_diffs)
    elif reduction == 'mean':
        return torch.tensor(distribution_diffs).mean()
    elif reduction == 'sum':
        return torch.tensor(distribution_diffs).sum()