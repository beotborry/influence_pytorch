import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

def split_dataset(features, labels, protected_attributes):
    '''
    list -> tensor
    '''
    z_groups = []
    t_groups = []
    for sen_arr in protected_attributes:
        z_group = []
        t_group = []

        for idx, sen_val in enumerate(sen_arr):
            if sen_val == 1:
                z_group.append(features[idx])
                t_group.append(labels[idx])

        z_groups.append(torch.FloatTensor(z_group))
        t_groups.append(torch.LongTensor(t_group))

    return z_groups, t_groups

def get_eopp_idx(t_groups):
    '''
    return idxs where true_label(y) is equal to 1
    return shape [[group0 idxs for y = 1], [group1 idxs for y = 1]]
    '''
    ret = []
    for t_group in t_groups:
        idx = np.where(t_group.numpy() == 1)
        ret.append(idx)
    return ret

def get_eo_idx(t_groups):
    '''
    return idxs where true_label(y) is equal to 0 and 1
    return shape [[group0 idxs for y = 0, idxs for y = 1], [group1 idxs for y = 0, idxs for y = 1]]
    '''
    ret = []
    for t_group in t_groups:
        tmp = []
        idx_0 = np.where(t_group.numpy() == 0)
        idx_1 = np.where(t_group.numpy() == 1)
        tmp.append(idx_0)
        tmp.append(idx_1)
        ret.append(tmp)
    return ret

def exp_normalize(x, scale_factor):
    x = -x * (scale_factor)
    b = x.max()
    y = np.exp(x - b)
    return (y / y.sum()) * len(x)

def calc_loss_diff(constraint, z_groups, t_groups, idxs, model):
    '''
    return violation for two groups (binary case)
    '''
    model.eval()

    if constraint == "eopp":
        losses = torch.zeros(len(z_groups))
        i = 0
        for z_group, t_group, idx in zip(z_groups, t_groups, idxs):
            losses[i] = nn.CrossEntropyLoss()(model(z_group[idx]), t_group[idx])
            i += 1
        return abs(losses[0] - losses[1])

    elif constraint == 'eo':
        losses = []
        for z_group, t_group, idx in zip(z_groups, t_groups, idxs):
            loss_0 = nn.CrossEntropyLoss()(model(z_group[idx[0]]), t_group[idx[0]])
            loss_1 = nn.CrossEntropyLoss()(model(z_group[idx[1]]), t_group[idx[1]])
            losses.append(loss_0)
            losses.append(loss_1)
        return max(abs(losses[0] - losses[2]), abs(losses[1]) - losses[3])

    elif constraint == 'dp':
        losses = []
        for z_group, t_group, idx in zip(z_groups, t_groups, idxs):
            loss_0 = nn.CrossEntropyLoss()(model(z_group[idx[0]]), t_group[idx[0]])
            loss_1 = nn.CrossEntropyLoss()(model(z_group[idx[1]]), t_group[idx[1]])
            losses.append(loss_0)
            losses.append(loss_1)
        return max(abs(losses[0] - losses[1]), abs(losses[2] - losses[3]))

def calc_fairness_metric(constraint, z_groups, t_groups, model):
    '''
    return fairness metric value for each fairness constraint with two groups(binary case)
    '''
    model.eval()

    confusion_matrix_groups = []

    for g in range(len(z_groups)):
        _X = z_groups[g]
        _y = t_groups[g]
        confusion_mat = confusion_matrix(_y.detach().numpy(),
                                         model(_X).argmax(dim=1).detach().numpy()).ravel()
        confusion_matrix_groups.append(confusion_mat)

    tn_0, fp_0, fn_0, tp_0 = confusion_matrix_groups[0]
    tn_1, fp_1, fn_1, tp_1 = confusion_matrix_groups[1]


    if constraint == 'eopp':
        return abs(tp_0 / (fn_0 + tp_0) - tp_1 / (fn_1 + tp_1))

    elif constraint == 'eo':
        return (abs(tp_0 / (fn_0 + tp_0) - tp_1 / (fn_1 + tp_1)) + abs(tn_0 / (fp_0 + tn_0) - tn_1 / (fp_0 + tn_1))) / 2.

    elif constraint == 'dp':
        return abs((fp_0 + tp_0) / (tn_0 + fp_0 + fn_0 + tp_0) - (fp_1 + tp_1) / (tn_1 + fp_1 + fn_1 + tp_1))


def get_error_and_violations(constraint, y_pred, y, protected_attributes):
    if constraint == 'eopp':
        acc = np.mean(y_pred != y)
        violations = []
        for p in protected_attributes:
            protected_idxs = np.where(np.logical_and(p > 0, y > 0))
            positive_idxs = np.where(y > 0)
            violations.append(np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs]))
        return acc, violations

    elif constraint == 'eo':
        acc = np.mean(y_pred != y)
        violations = []
        for p in protected_attributes:
            protected_idxs = np.where(np.logical_and(p > 0, y > 0))
            positive_idxs = np.where(y > 0)
            violations.append(np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs]))
            protected_idxs = np.where(np.logical_and(p > 0, y < 1))
            negative_idxs = np.where(y < 1)
            violations.append(np.mean(y_pred[negative_idxs]) - np.mean(y_pred[protected_idxs]))
        return acc, violations

    elif constraint == 'dp':
        acc = np.mean(y_pred != y)
        violations = []
        for p in protected_attributes:
            protected_idxs = np.where(p > 0)
            violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
        return acc, violations


def debias_weights(constraint, original_labels, protected_attributes, multipliers):
    if constraint == 'eopp':
        exponents = np.zeros(len(original_labels))
        for i, m in enumerate(multipliers):
            exponents -= m * protected_attributes[i]
        weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
        weights = np.where(original_labels > 0, 1 - weights, weights)
        return weights

    elif constraint == 'eo':
        exponents_pos = np.zeros(len(original_labels))
        exponents_neg = np.zeros(len(original_labels))

        for i, protected in enumerate(protected_attributes):
            exponents_pos -= multipliers[2 * i] * protected
            exponents_neg -= multipliers[2 * i + 1] * protected
        weights_pos = np.exp(exponents_pos) / (np.exp(exponents_pos) + np.exp(-exponents_pos))
        weights_neg = np.exp(exponents_neg) / (np.exp(exponents_neg) + np.exp(-exponents_neg))

        # weights = np.where(predicted > 0, weights, 1 - weights)
        weights = np.where(original_labels > 0, 1 - weights_pos, weights_neg)
        return weights

    elif constraint == 'dp':
        exponents = np.zeros(len(original_labels))
        for i, m in enumerate(multipliers):
            exponents -= m * protected_attributes[i]
        weights = np.exp(exponents) / (np.exp(exponents) + np.exp(-exponents))
        weights = np.where(original_labels > 0, 1 - weights, weights)
        return weights





