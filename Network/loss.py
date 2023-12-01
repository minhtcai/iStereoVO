import torch
import torch.nn as nn

def motion_loss_fn(motions, gt_motions):
    """
    Compute the loss.
    """
    # Normalize the GT motion translations. Those are already normalized in the network for the predicted translations.
    # motions[:, 0:3] = motions[:, 0:3].clone() / torch.linalg.norm(motions[:, 0:3].clone(), dim=1, keepdim=True)
    # gt_motions[:, 0:3] = gt_motions[:, 0:3].clone() / torch.linalg.norm(gt_motions[:, 0:3].clone(), dim=1, keepdim=True)

    # Compute the loss.
    # For the translation, we care about the colinearity of the vectors, so we use cosine similarity. It is true that upon a zero-motion, the cosine similarity will be undefined, but in practice this is not a problem.
    loss_trans = 1 - nn.functional.cosine_similarity(motions[:, 0:3], gt_motions[:, 0:3], dim=1).mean()

    # Rotation. We use MSE loss.
    loss_rot = torch.nn.functional.mse_loss(motions[:, 3:7], gt_motions[:, 3:7], reduction='mean')

    return loss_trans, loss_rot


def flow_loss_fn(flow_pred, flow_gt, valid,
                gamma=0.9,
                max_flow=400,
                **kwargs,
                ):
    '''
    Flow loss function from GMFlow. Operating on flow in a single scale.
    Args:
        flow_preds (list of torch tensor B, H, W, 2): list of predicted flows 
        flow

        flow_gt: ground truth flow
        valid: valid mask
        gamma: gamma for weighted loss

    '''
    n_predictions = 1
    flow_loss = 0.0

    # exclude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)

    # l1 loss for the flow.
    loss = (flow_pred - flow_gt).abs()

    # # If covariance is provided, use it to weight the loss.
    # if 'cov' in kwargs:
    #     cov = kwargs['cov']
    #     loss = loss * torch.exp(-cov)

    # Apply the mask.
    flow_loss = (valid[:, None] * loss).mean()

    # Compute the EPE. End point error.
    epe = torch.sum((flow_pred - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(), # Percentage of pixels with EPE > 1.
        '3px': (epe > 3).float().mean().item(), # Percentage of pixels with EPE > 3.
        '5px': (epe > 5).float().mean().item(), # Percentage of pixels with EPE > 5.
    }

    return flow_loss, metrics