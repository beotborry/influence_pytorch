from compas_dataloader import get_data
import torch
from torch.optim import SGD, Adam
from utils import split_dataset_multi, exp_normalize, get_eopp_idx
from mlp import MLP
from influence_function import avg_s_test_multi
from tqdm import tqdm
import torch.nn as nn
from influence_function import calc_influence
from adult_dataloader import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, protected_train, protected_test = get_data()

    z_groups_train, t_groups_train = split_dataset_multi(X_train, y_train, protected_train)
    z_groups_test, t_groups_test = split_dataset_multi(X_test, y_test, protected_test)

    eopp_idx_train = get_eopp_idx(t_groups_train)
    eopp_idx_test = get_eopp_idx(t_groups_test)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    num_features = 31

    model = MLP(
        feature_size=num_features,
        hidden_dim=50,
        num_classes=2,
        num_layer=2
    )
    #optimizer = Adam(model.parameters(), lr=0.001)
    optimizer = SGD(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max')
    criterion = nn.CrossEntropyLoss(reduction = 'none')

batch_size = 128

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

gpu = 1 if torch.cuda.is_available() else -1
print(gpu)

def calc_influence_dataset(X, y, z_groups, t_groups,
                           model, train_loader, gpu):

    s_test_vec = avg_s_test_multi(z_groups=z_groups, t_groups=t_groups, idxs = eopp_idx_train, model=model, z_loader=train_loader, gpu=gpu, r=3)

    influences = []
    for z, t in zip(X, y):
        influences.append(calc_influence(z, t, s_test_vec, model, train_loader, gpu=gpu))
    return influences

iteration = 30

max_iter = 0
max_tradeoff = 0


for iter in range(iteration):
    print("Iteration: {}".format(iter))
    if iter == 0: weights = torch.ones(len(X_train))
    else:
        weights = torch.tensor(calc_influence_dataset(X_train, y_train, z_groups_train, t_groups_train,
                                                      model, train_loader, gpu = gpu))
    weights = exp_normalize(weights)
    print(weights)
    for epoch in tqdm(range(20)):
        model.train()
        i = 0
        for z, t in train_loader:
            if gpu >= 0: z, t, model = z.cuda(), t.cuda(), model.cuda()
            model.train()
            y_pred = model(z)

            weight = weights[i * batch_size: (i+1) * batch_size] if (i+1) * batch_size <= len(X_train) else weights[i*batch_size:]
            if gpu >= 0: weight = weight.cuda()

            loss = torch.mean(weight * criterion(y_pred, t))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i = i + 1

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

    violation = abs(group_loss[0] - group_loss[1])

    print("Iteration {}, Violation: {}".format(iter, violation))

    group_confusion_matrix = []

    for g in range(len(protected_test)):
        _X = z_groups_test[g]
        _y = t_groups_test[g]
        confusion_mat = confusion_matrix(_y.detach().numpy(),
                                         model(_X).argmax(dim=1).detach().numpy()).ravel()
        group_confusion_matrix.append(confusion_mat)

    eopp_metrics = 0
    for i in range(len(group_confusion_matrix)):
        for j in range(i + 1, len(group_confusion_matrix)):
            _, _, fn_1, tp_1 = group_confusion_matrix[i]
            _, _, fn_2, tp_2 = group_confusion_matrix[j]

            eopp_metrics = abs(tp_1 / (fn_1 + tp_1) - tp_2 / (fn_2 + tp_2))

    print("Iteration: {}, Eopp: {}".format(iter, eopp_metrics))

    _tradeoff = accuracy / eopp_metrics

    if _tradeoff > max_tradeoff and iter >= 3:
        max_iter = iter
        max_tradeoff = _tradeoff
        torch.save(model, "model/compas_influence_best2")

print("max_iter:{}, max_tradeoff:{}".format(max_iter, max_tradeoff))





