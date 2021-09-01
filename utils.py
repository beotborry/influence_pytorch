import numpy as np
import torch.nn.functional as F
import torch

def split_dataset(features, labels, sen_attrs, constraint):
    # _(labels)(group idx)

    if constraint == "EO":
        features_0 = []
        labels_0 = []
        features_1 = []
        labels_1 = []

        for idx, sen_val in enumerate(sen_attrs):
            if sen_val == 1 and labels[idx] == 0:
                features_0.append(features[idx])
                labels_0.append(labels[idx])
            elif sen_val == 1 and labels[idx] == 1:
                features_1.append(features[idx])
                labels_1.append(labels[idx])
        return [features_0, features_1], [labels_0, labels_1]

    elif constraint == "EOpp":
        features_1 = []
        labels_1 = []
        for idx, sen_val in enumerate(sen_attrs):
            if sen_val == 1 and labels[idx] == 1:
                features_1.append(features[idx])
                labels_1.append(labels[idx])
        return features_1, labels_1

    else:
        ret_features = []
        ret_labels = []
        for idx, sen_val in enumerate(sen_attrs):
            if sen_val == 1:
                ret_features.append(features[idx])
                ret_labels.append(labels[idx])

        return ret_features, ret_labels


def logloss_one_label(true_label, predicted, eps=1e-15, softmax=False):
    #p = np.clip(predicted, eps, 1-eps)
    if softmax:
        predicted = F.softmax(predicted, dim=1)


    if true_label == 1:
        predicted = predicted[:, 1]
        return torch.sum(-torch.log(predicted))
    else:
        predicted = predicted[:, 0]
        return -torch.log(1-predicted)

def exp_normalize(x):
    x = x * (10 ** 4)
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()