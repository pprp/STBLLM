import torch

from utils.autosearch import structural_searching
from utils.mask import generate_structural_mask


def fun1_standardize(M):
    # M can be weight or activation or gradient
    M_mean = M.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
    M = M - M_mean
    std = M.view(M.size(0), -1).std(dim=1).view(-1, 1) + 1e-5
    M = M / std.expand_as(M)
    return M


# 4mask
def structural_guassian_distribution(
    tmp, H=None, X_dict=None, metric='magnitude', up_lim=30, engine=None
):
    if metric == 'hessian':
        target_weights = tmp**2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == 'magnitude':
        target_weights = tmp
    elif metric == 'ria':
        X = X_dict['ROW']
        target_weights = (
            torch.abs(tmp) / torch.sum(torch.abs(tmp), dim=0)
            + torch.abs(tmp) / torch.sum(torch.abs(tmp), dim=1).reshape(-1, 1)
        ) * (torch.sqrt(X)) ** 0.5
        target_weights = fun1_standardize(target_weights)
    elif metric == 'auto':
        target_weights = engine.compute_metric(tmp, X_dict)
    else:
        raise NotImplementedError

    optimal_split_1, optimal_split_2, mask4 = structural_searching(
        target_weights, up_lim
    )
    mask1, mask2, mask3 = generate_structural_mask(
        target_weights, mask4, optimal_split_1, optimal_split_2
    )

    print(
        (mask1.sum() / mask1.numel()).item(),
        (mask2.sum() / mask2.numel()).item(),
        (mask3.sum() / mask3.numel()).item(),
        (mask4.sum() / mask4.numel()).item(),
    )
    return mask1, mask2, mask3, mask4
