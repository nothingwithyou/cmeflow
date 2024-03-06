import torch
import torch.nn.functional as F
from munch import Munch
from .utils import FieldWarper

device = 'cuda'


def flow_loss_func(flow_preds, flow_gt, valid,
                   gamma=0.9,
                   max_flow=400,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        i_loss = (flow_preds[i] - flow_gt).abs()

        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss, metrics


loss_registry = {}


def charbonnier_loss(x, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss for x
    Args:
        x(tesnor): [batch, channels, height, width]
    Returns:
        loss
    """
    batch, channels, height, width = x.shape
    normalization = torch.tensor(batch * height * width * channels,
                                 requires_grad=False)

    error = torch.pow(
        (x * torch.tensor(beta)).pow(2) + torch.tensor(epsilon).pow(2), alpha)

    return torch.sum(error) / normalization


def _warp_deltas(photo):
    """1st order smoothness, compute smoothness loss components"""
    weight = construct_gradient_kernel(device)
    pad_photo = torch.nn.ReplicationPad2d(1)(photo)
    delta_u = F.conv2d(pad_photo, weight)
    delta_v = F.conv2d(pad_photo, weight)
    return delta_u + delta_v


def warp_loss(img_first, img_second, flow):
    """Differentiable Charbonnier penalty function"""

    difference = img_second - FieldWarper.backward_warp(tensorInput=img_first,
                                                        tensorFlow=flow) + _warp_deltas(img_second) - _warp_deltas(
        FieldWarper.backward_warp(tensorInput=img_first,
                                  tensorFlow=flow))
    return charbonnier_loss(difference, beta=1000.0)
    # return charbonnier_loss(difference, beta=255.0)


def construct_divergence_kernels():
    """
    Construct the x and y grad kernel seperately, typically for computing divergence use
    Returns: 1st order kernels for both x and y direction grads

    """
    filter_x = torch.ones(1, 1, 3, 3, requires_grad=False)
    filter_y = torch.ones(1, 1, 3, 3, requires_grad=False)
    filter_x[0, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    filter_y[0, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
    return filter_x, filter_y


def construct_gradient_kernel(device="cuda"):
    out_channels = 2  # u and v
    in_channels = 1  # u or v
    kh, kw = 3, 3

    filter_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
    weight[0, 0, :, :] = filter_x
    weight[1, 0, :, :] = filter_y
    return weight.to(device)


def construct_laplace_kernel(device="cuda"):
    out_channels = 2  # u and v
    in_channels = 1  # u or v
    kh, kw = 3, 3

    filter_x = torch.FloatTensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
    filter_y = torch.FloatTensor([[0, 1, 0], [0, -2, 0], [0, 1, 0]])

    weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
    weight[0, 0, :, :] = filter_x
    weight[1, 0, :, :] = filter_y
    return weight.to(device)


def construct_forward_difference_kernels(device="cuda"):
    filter_x = torch.ones(1, 1, 3, 3, requires_grad=False)
    filter_y = torch.ones(1, 1, 3, 3, requires_grad=False)
    filter_x[0, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    filter_y[0, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    return filter_x.to(device), filter_y.to(device)


def _smoothness_deltas(flow):
    """1st order smoothness, compute smoothness loss components"""
    out_channels = 2  # u and v
    in_channels = 1  # u or v
    kh, kw = 3, 3

    filter_x = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    filter_y = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])

    weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
    weight[0, 0, :, :] = filter_x
    weight[1, 0, :, :] = filter_y

    uFlow, vFlow = torch.split(flow, split_size_or_sections=1, dim=1)

    delta_u = F.conv2d(uFlow, weight.to(device))
    delta_v = F.conv2d(vFlow, weight.to(device))
    return delta_u, delta_v


def laplace_operator(flow):
    laplace_kernel = construct_laplace_kernel(device)
    u, v = torch.split(flow, split_size_or_sections=1, dim=1)
    laplace_u = F.conv2d(u, laplace_kernel, padding=1)
    laplace_v = F.conv2d(v, laplace_kernel, padding=1)
    return laplace_u, laplace_v


def gradient_operator(flow):
    gradient_kernel = construct_gradient_kernel(device)
    u, v = torch.split(flow, split_size_or_sections=1, dim=1)
    grad_u = F.conv2d(u, gradient_kernel, padding=1)
    grad_v = F.conv2d(v, gradient_kernel, padding=1)
    return grad_u, grad_v


def smoothness_loss(flow):
    """Compute 1st order smoothness loss"""
    delta_u, delta_v = _smoothness_deltas(flow)
    return charbonnier_loss(delta_u) + charbonnier_loss(delta_v)


def curl_operator(flow):
    """
    Compute the curl of 2D flow field
    """
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    filter_x, filter_y = construct_divergence_kernels()
    grad_u = F.conv2d(flow_u, filter_y.to(device), padding=1)
    grad_v = F.conv2d(flow_v, filter_x.to(device), padding=1)
    return grad_v - grad_u


def divergence_loss(flow):
    """
    Compute the divergence loss
    """
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    filter_x, filter_y = construct_divergence_kernels()
    grad_u = F.conv2d(flow_u, filter_x.to(device), padding=1)
    grad_v = F.conv2d(flow_v, filter_y.to(device), padding=1)
    return charbonnier_loss(grad_u + grad_v)


def unsupervised_error(tensorFlowForward,
                       tensorFlowBackward,
                       tensorFirst,
                       tensorSecond,
                       args,
                       gamma=0.9,
                       max_flow=400,
                       curl_loss=None,
                       ):
    h, w = tensorFirst.shape[-2], tensorFirst.shape[-1]

    loss_photowarp = 0
    loss_biflow = 0
    loss_smooth = 0
    loss_div = 0
    n_predictions = len(tensorFlowForward)

    # exlude invalid pixels and extremely large diplacements

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        local_forward = tensorFlowForward[i]
        local_backward = tensorFlowBackward[i]
        bi_warp = warp_loss(tensorFirst, tensorSecond, local_forward) + \
                  warp_loss(tensorSecond, tensorFirst, local_backward)
        loss_photowarp += bi_warp * i_weight
        loss_smooth += smoothness_loss(local_forward) * i_weight
        loss_biflow += F.l1_loss(local_forward, -local_backward) * i_weight
        loss_div += divergence_loss(local_forward) * i_weight
    if curl_loss is None:
        loss = loss_photowarp * args.lambda_photowarp + loss_biflow * args.lambda_biflow + \
               loss_smooth * args.lambda_smooth + loss_div * args.lambda_div
        return loss, Munch(photo=loss_photowarp.item(),
                           bi=loss_biflow.item(),
                           sm=loss_smooth.item(),
                           div=loss_div.item(),
                           total=loss.item())
    else:
        loss = loss_photowarp * args.lambda_photowarp + loss_biflow * args.lambda_biflow + \
               loss_smooth * args.lambda_smooth + loss_div * args.lambda_div + curl_loss * args.lambda_ts
        return loss, Munch(photo=loss_photowarp.item(),
                           bi=loss_biflow.item(),
                           sm=loss_smooth.item(),
                           div=loss_div.item(),
                           curl_loss=curl_loss.item(),
                           total=loss.item())


