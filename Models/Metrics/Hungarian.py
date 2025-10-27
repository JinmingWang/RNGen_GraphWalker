from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as func

Tensor = torch.Tensor


def hungarianMetric(batch_pred_edges, batch_target_edges, reduction: str = "mean"):
    """
    Find hungarian matching between predicted and target segments.
    Then compute MAE and MSE between matched segments.

    Hungarian matching is used to find global optimal matching between predicted and target segments.

    :param batch_pred_edges: list of predicted segments, each: (N_edges, N_interp, 2)
    :param batch_target_edges: list of target segments
    :param reduction: reduction method for MAE and MSE, "none", "mean" or "sum"
    :return: MAE, MSE
    """

    B = len(batch_pred_edges)
    mae_list = []
    mse_list = []

    for b in range(B):
        pred_edges = batch_pred_edges[b].flatten(1)  # (P, N_interp*2)
        target_edges = batch_target_edges[b].flatten(1)  # (Q, N_interp*2)
        target_edges_flip = batch_target_edges[b].flip(1).flatten(1)  # (Q, N_interp*2)
        P = len(pred_edges)
        Q = len(target_edges)

        # if P > Q:
        #     target_edges = func.pad(target_edges, (0, 0, 0, P - Q))
        #     target_edges_flip = func.pad(target_edges_flip, (0, 0, 0, P - Q))
        # elif Q > P:
        #     pred_edges = func.pad(pred_edges, (0, 0, 0, Q - P))

        cost_matrix = torch.cdist(pred_edges, target_edges, p=2)  # (P, Q)
        cost_matrix_flip = torch.cdist(pred_edges, target_edges_flip, p=2)  # (P, Q)
        cost_matrix = torch.minimum(cost_matrix, cost_matrix_flip).cpu().detach().numpy()

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Matched segments, compute MAE and MSE normally for matched segments
        matched_pred_edges = pred_edges[row_ind]
        matched_target_edges = target_edges[col_ind]

        mae = torch.abs(matched_pred_edges - matched_target_edges).mean().item()
        mse = ((matched_pred_edges - matched_target_edges) ** 2).mean().item()

        n_mismatch = abs(P - Q)
        mismatch_mae = n_mismatch * 1.0     # assume matching (-2, -2) to (2, 2), MAE = (|-2-2|+|-2-2|)/2 = 4
        mismatch_mse = n_mismatch * 1.0

        mae_list.append(mae + mismatch_mae)
        mse_list.append(mse + mismatch_mse)

    if reduction == "none":
        return torch.tensor(mae_list), torch.tensor(mse_list)
    elif reduction == "mean":
        return torch.tensor(mae_list).mean().item(), torch.tensor(mse_list).mean().item()
    elif reduction == "sum":
        return torch.tensor(mae_list).sum().item(), torch.tensor(mse_list).sum().item()

