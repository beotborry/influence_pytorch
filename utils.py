import numpy as np
import torch.nn.functional as F
import torch

def split_dataset(features, labels, sen_attrs, constraint=None):
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

def exp_normalize(x):
    x = -x * (500)
    b = x.max()
    y = np.exp(x - b)
    return (y / y.sum()) * len(x)

def get_accuracy(outputs, labels, binary=False, sigmoid_output=False, reduction='mean'):
    #if multi-label classification
    if len(labels.size())>1:
        outputs = (outputs>0.0).float()
        correct = ((outputs==labels)).float().sum()
        total = torch.tensor(labels.shape[0] * labels.shape[1], dtype=torch.float)
        avg = correct / total
        return avg.item()
    if binary:
        if sigmoid_output:
            predictions = (outputs >= 0.5).float()
        else:
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
    else:
        predictions = torch.argmax(outputs, 1)
    c = (predictions == labels).float().squeeze()

    if reduction == 'none':
        return c
    else:
        accuracy = torch.mean(c)
        return accuracy.item()