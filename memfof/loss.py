import torch

# exclude extremly large displacements
MAX_FLOW = 512


def sequence_loss(
    outputs: dict[str, list[torch.Tensor]],
    flow_gts: torch.Tensor,
    valids: torch.Tensor,
    gamma: float = 0.8,
    max_flow: float = MAX_FLOW,
) -> torch.Tensor:
    """Calculate sequence loss for optical flow estimation.

    Parameters
    ----------
    outputs : Dict[str, List[torch.Tensor]]
        Dictionary containing model outputs:
        - 'flow': List of predicted flow fields, each of shape (B, 2, 2, H, W)
        - 'nf': List of normalized flow losses, each of shape (B, 2, 2, H, W)
    flow_gts : torch.Tensor
        Ground truth flow fields of shape (B, 2, 2, H, W)
    valids : torch.Tensor
        Validity masks of shape (B, 2, H, W)
    gamma : float, optional
        Weight decay factor for sequence loss, by default 0.8
    max_flow : float, optional
        Maximum flow magnitude threshold, by default MAX_FLOW

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """
    n_predictions = len(outputs["flow"])
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gts**2, dim=2).sqrt()
    valid = (valids >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = outputs["nf"][i]
        final_mask = (
            (~torch.isnan(loss_i.detach()))
            & (~torch.isinf(loss_i.detach()))
            & valid[:, :, None]
        )

        fms = final_mask.sum()
        if fms > 0.5:
            flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())
        else:
            flow_loss += (0.0 * loss_i).sum().nan_to_num(0.0)

    return flow_loss
