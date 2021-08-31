from utils import split_dataset, logloss_one_label
from torch.autograd import grad
import torch
import torch.nn.functional as F
import torch.nn as nn
from influence_function_utils import display_progress
import numpy as np

def grad_z(z, t, model, gpu=-1):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.
    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, feature_size)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU
    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""

    if z.dim() == 1: z = z.view(1, z.shape[0])
    if t.dim() != 1: t = t.view(1)

    model.eval()
    if gpu >= 0:
        z, t = z.cuda(), t.cuda()
    y = model(z)
    y = F.softmax(y)
    if y.dim() == 1: y = y.view(1, 2)

    loss = F.cross_entropy(y,t)

    params = [p for p in model.parameters() if p.requires_grad]
    return list(grad(loss, params, create_graph=True))

def s_test(z_group1, t_group1, z_group2, t_group2, model, z_loader, recursion_depth=5000, damp=0.01, scale=25.0, gpu=-1):
    model.eval()

    group1_loss = nn.CrossEntropyLoss()(model(z_group1), t_group1)
    group2_loss = nn.CrossEntropyLoss()(model(z_group2), t_group2)

    violation = group1_loss - group2_loss

    params = [p for p in model.parameters() if p.requires_grad]
    v = list(grad(violation, params, create_graph=True))
    h_estimate = v.copy()
    for i in range(recursion_depth):
        for x, t in z_loader:
            if gpu >= 0:
                x, t = x.cuda(), t.cuda()
            y = model(x)
            y = F.softmax(y)
            loss = torch.nn.functional.cross_entropy(y, t)
            hv = hvp(loss, params, h_estimate)
            with torch.no_grad():
                h_estimate = [
                    _v + (1 - damp) * _h_e - _hv / scale
                    for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
    return h_estimate

def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.
    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop

    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads

def calc_influence(z, t, s_test, model, z_loader):

    s_test_vec = s_test
    grad_z_vec = grad_z(z = z, t = t, model = model)

    influence = -sum([
        torch.sum(k * j).data for k, j in zip(grad_z_vec, s_test_vec)
    ]) / len(z_loader.dataset)

    return influence