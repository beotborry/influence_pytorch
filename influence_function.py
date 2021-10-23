from torch.autograd import grad
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import calc_loss_diff


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
    if y.dim() == 1: y = y.view(1, 2)

    loss = torch.nn.CrossEntropyLoss()(y,t)

    params = [p for p in model.parameters() if p.requires_grad]
    return list(grad(loss, params, create_graph=True))

def s_test(z_groups, t_groups, idxs, model, z_loader, constraint, weights, recursion_depth=5000, damp=0.01, scale=25.0, gpu=-1):
    model.eval()
    violation = calc_loss_diff(constraint, z_groups, t_groups, idxs, model)

    params = [p for p in model.parameters() if p.requires_grad]
    v = list(grad(violation, params, create_graph=True))
    h_estimate = v.copy()
    for x, t, idx in z_loader:
        if gpu >= 0:
            x, t = x.cuda(), t.cuda()
        y = model(x)
        #y = F.softmax(y, dim=0)

        loss = weights[idx] * torch.nn.CrossEntropyLoss(reduction='none')(y, t)
        #loss = weights[idx].detach() * torch.nn.CrossEntropyLoss(reduction='none')(y,t)
        #loss = torch.nn.functional.cross_entropy(y, t)
        #loss = weights[idx] * torch.nn.CrossEntropyLoss(reduction='none')(y, t)
        break
    for i in range(recursion_depth):
        hv = hvp(loss[i], params, h_estimate)
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
    return h_estimate

def avg_s_test(z_groups, t_groups, idxs, model, z_loader, constraint, weights, recursion_depth=5000, damp=0.01, scale=25.0, gpu=-1, r=1):
    all = s_test(z_groups=z_groups, t_groups=t_groups, idxs=idxs, model=model, z_loader=z_loader,
                                      constraint=constraint, weights = weights,
                                      recursion_depth=recursion_depth, damp=damp, scale=scale, gpu=gpu)

    for i in range(1, r):
        cur = s_test(z_groups=z_groups, t_groups=t_groups, idxs=idxs, model=model, z_loader=z_loader,
                                      constraint=constraint, weights = weights,
                                      recursion_depth=recursion_depth, damp=damp, scale=scale, gpu=gpu)
        all = [a + c for a, c in zip(all, cur)]

    all = [a / r for a in all]
    return all


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

def calc_influence(z, t, s_test, model, z_loader, gpu = -1):

    s_test_vec = s_test
    grad_z_vec = grad_z(z = z, t = t, model = model, gpu = gpu)
    influence = -sum([
        torch.sum(k * j).data for k, j in zip(grad_z_vec, s_test_vec)
    ]) / len(z_loader.dataset)

    return influence

def calc_influence_dataset(X, y, idxs, z_groups, t_groups, model, z_loader, weights, gpu, constraint, r=1, recursion_depth=5000, scale=25.0):
    s_test_vec = avg_s_test(z_groups=z_groups, t_groups=t_groups, idxs=idxs, model=model, z_loader=z_loader, gpu=gpu,
                            constraint=constraint, weights=weights, scale=scale,
                            r=r, recursion_depth=recursion_depth)
    influences = []
    for z, t in zip(X, y):
        influences.append(calc_influence(z, t, s_test_vec, model, z_loader, gpu=gpu))
    return influences