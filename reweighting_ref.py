import torch
from adult_dataloader import get_data
from adult_dataloader import CustomDataset
from utils import split_dataset_multi, get_eopp_idx
import numpy as np
from mlp import MLP
from torch.optim import SGD
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import time



X_train, y_train, X_test, y_test, protected_train, protected_test = get_data()

z_groups_train, t_groups_train = split_dataset_multi(X_train, y_train, protected_train)
z_groups_test, t_groups_test = split_dataset_multi(X_test, y_test, protected_test)

eopp_idx_train = get_eopp_idx(t_groups_train)
eopp_idx_test = get_eopp_idx(t_groups_test)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

num_features = X_train.shape[1]

model = MLP(
    feature_size=num_features,
    hidden_dim=50,
    num_classes=2,
    num_layer=2
)

optimizer = SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(reduction = 'none')

batch_size = 128

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

gpu = 1 if torch.cuda.is_available() else -1
print(gpu)

def get_error_and_violations(y_pred, y, protected_attributes):
    acc = np.mean(y_pred != y)
    violations = []
    for p in protected_attributes:
        protected_idxs = np.where(np.logical_and(p > 0, y > 0))
        positive_idxs = np.where(y > 0)
        violations.append(np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs]))
    return acc, violations


def debias_weights(original_labels, protected_attributes, multipliers):
    exponents = np.zeros(len(original_labels))
    for i, m in enumerate(multipliers):
        exponents -= m * protected_attributes[i]
    weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
    #weights = np.where(predicted > 0, weights, 1 - weights)
    weights = np.where(original_labels > 0, 1 - weights, weights)
    return weights

iteration = 30
epochs = 20
max_iter = 0
max_tradeoff = 0

multipliers = np.zeros(len(protected_train))
weights = np.array([1] * X_train.shape[0])
eta = 1.

start = time.time()
for iter in range(iteration):
    print("Iteration: {}".format(iter))
    if iter == 0: weights = torch.tensor(np.array([1] * X_train.shape[0]))
    else: weights = torch.tensor(debias_weights(y_train, protected_train, multipliers))
    for epoch in tqdm(range(epochs)):
        model.train()
        i = 0
        for z, t in train_loader:
            if gpu >= 0: z, t, model = z.cuda(), t.cuda(), model.cuda()
            y_pred = model(z)
            weight = weights[i * batch_size: (i + 1) * batch_size] if (i + 1) * batch_size <= len(X_train) else weights[i * batch_size:]
            if gpu >= 0: weight = weight.cuda()

            loss = torch.mean(weight * criterion(y_pred, t))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1

    y_pred_train = model(X_train).argmax(dim = 1).detach().numpy()
    _, violations = get_error_and_violations(y_pred_train, y_train, protected_train)
    multipliers += eta * np.array(violations)

    model.eval()
    print()
    with torch.no_grad():
        accuracy = 0
        i = 0
        for z, t in test_loader:
            if gpu >= 0: z, t = z.cuda(), t.cuda()
            y_pred = model(z)
            accuracy += (((y_pred.argmax(dim=1) == t).sum()) / len(t)).item()
            i = i + 1

        print("Iteration {}, Test Acc: {}".format(iter, accuracy / i))

    if gpu >= 0: z_groups_train, t_groups_train = z_groups_train.cuda(), t_groups_train.cuda()

    group_loss = []
    for g in range(len(protected_test)):
        _X = z_groups_test[g]
        _y = t_groups_test[g]
        _idx = eopp_idx_test[g]

        group_loss.append(nn.CrossEntropyLoss()(model(_X[_idx]), _y[_idx]))

    violation = 0
    for i in range(len(group_loss) - 1):
        _violation = abs(group_loss[i] - group_loss[i + 1])
        violation = max(_violation, violation)

    print("Iteration {}, Violation: {}".format(iter, violation))

    group_confusion_matrix = []

    for g in range(len(protected_test)):
        _X = z_groups_test[g]
        _y = t_groups_test[g]
        confusion_mat = confusion_matrix(_y.detach().numpy(),
                                         model(_X).argmax(dim=1).detach().numpy()).ravel()
        group_confusion_matrix.append(confusion_mat)

    eopp_metrics = []
    for i in range(len(group_confusion_matrix)):
        for j in range(len(group_confusion_matrix)):
            if i != j:
                _, _, fn_1, tp_1 = group_confusion_matrix[i]
                _, _, fn_2, tp_2 = group_confusion_matrix[j]

                _metrics = abs(tp_1 / (fn_1 + tp_1) - tp_2 / (fn_2 + tp_2))
                eopp_metrics.append(_metrics)

    eopp_metric = max(eopp_metrics)

    print("Iteration: {}, Eopp: {}".format(iter, eopp_metric))

    _tradeoff = accuracy / eopp_metric

    if _tradeoff > max_tradeoff:
        max_iter = iter
        max_tradeoff = _tradeoff
        #torch.save(model, "./model/bank_reweighting_best")

print("max_iter: {}, max_tradeoff: {}".format(max_iter, max_tradeoff))
end = time.time()
print((end - start) / 60)



