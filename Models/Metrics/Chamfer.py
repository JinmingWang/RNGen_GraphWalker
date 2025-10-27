import torch

Tensor = torch.Tensor


def chamferMetric(batch_pred_edges, batch_target_edges, reduction="mean"):
    """
    Find chamfer matching between predicted and target segments.
    Then compute MAE and MSE between matched segments.

    Chamfer matching is to find the best match for each segment in the predicted segments.

    :param batch_pred_edges: list of predicted segments
    :param batch_target_edges: list of target segments
    :param reduction: reduction method, "none", "mean", "sum"
    :return: Chamfer distance
    """
    B = len(batch_pred_edges)
    mae_list = []
    mse_list = []

    for b in range(B):
        pred_edges = batch_pred_edges[b].flatten(1)  # (P, N_interp*2)
        target_edges = batch_target_edges[b].flatten(1)  # (Q, N_interp*2)
        target_edges_flip = batch_target_edges[b].flip(1).flatten(1)  # (Q, N_interp*2)

        P, Q = len(pred_edges), len(target_edges)

        # if len(target_edges) == 0:
        #     target_edges = torch.zeros_like(pred_edges)
        #     target_edges_flip = torch.zeros_like(pred_edges)

        # Compute pairwise distance matrix
        cost_matrix = torch.cdist(pred_edges, target_edges, p=2)  # (P, Q)
        cost_matrix_flip = torch.cdist(pred_edges, target_edges_flip, p=2)
        cost_matrix = torch.minimum(cost_matrix, cost_matrix_flip)

        # Find the best match for each segment in the predicted segments
        row_ind_p2t = torch.argmin(cost_matrix, dim=1)

        # Compute MAE and MSE (match the predicted segments to the target segments)
        mae_p2t = (torch.abs(pred_edges - target_edges[row_ind_p2t]).mean())
        mse_p2t = ((pred_edges - target_edges[row_ind_p2t]) ** 2).mean()

        # Compute MAE and MSE (match the target segments to the predicted segments)
        row_ind_t2p = torch.argmin(cost_matrix, dim=0)
        mae_t2p = (torch.abs(pred_edges[row_ind_t2p] - target_edges).mean())
        mse_t2p = ((pred_edges[row_ind_t2p] - target_edges) ** 2).mean()

        # Why do we compute p2t and also t2p?
        # When we match p to t, some segments in p may not have a match in t, they are not counted in the loss.
        # When we match t to p, some segments in t may not have a match in p, they are not counted in the loss.
        # So we need to compute both to make sure all segments are counted in the loss.
        mae = (mae_p2t + mae_t2p).item()
        mse = (mse_p2t + mse_t2p).item()

        n_mismatch = abs(P - Q)
        mismatch_mae = n_mismatch * 1.0     # assume matching (-2, -2) to (2, 2), MAE = (|-2-2|+|-2-2|)/2 = 4
        mismatch_mse = n_mismatch * 1.0   # assume matching (-2, -2) to (2, 2), MSE = ((-2-2)^2+(-2-2)^2)/2 = (16+16)/2 = 16

        mae_list.append(mae + mismatch_mae)
        mse_list.append(mse + mismatch_mse)

    if reduction == "none":
        return torch.tensor(mae_list), torch.tensor(mse_list)
    elif reduction == "mean":
        return torch.tensor(mae_list).mean().item(), torch.tensor(mse_list).mean().item()
    elif reduction == "sum":
        return torch.tensor(mae_list).sum().item(), torch.tensor(mse_list).sum().item()
