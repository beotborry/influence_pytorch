import torch
import torch.nn as nn
import numpy as np
from utils import split_dataset, exp_normalize, calc_loss_diff, calc_fairness_metric, get_error_and_violations, debias_weights
from torch.optim import SGD
from mlp import MLP
from adult_dataloader import CustomDataset
from torch.utils.data import DataLoader
from influence_function import calc_influence_dataset
from tqdm import tqdm
from argument import get_args
from time import time

def main():
    args = get_args()

    dataset = args.dataset
    fairness_constraint = args.constraint
    method = args.method

    if dataset == "adult":
        from adult_dataloader import get_data
    elif dataset == "bank":
        from bank_dataloader import get_data
    elif dataset == "compas":
        from compas_dataloader import get_data

    X_train, y_train, X_test, y_test, protected_train, protected_test = get_data()

    X_groups_train, y_groups_train = split_dataset(X_train, y_train, protected_train)
    X_groups_test, y_groups_test = split_dataset(X_test, y_test, protected_test)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    batch_size = 128

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    gpu = -1 if torch.cuda.is_available() else -1

    if fairness_constraint == 'eopp':
        from utils import get_eopp_idx
        get_idx = get_eopp_idx
    elif fairness_constraint == 'eo':
        from utils import get_eo_idx
        get_idx = get_eo_idx
    elif fairness_constraint == 'dp':
        from utils import get_eo_idx
        get_idx = get_eo_idx

    constraint_idx_train = get_idx(y_groups_train)
    constraint_idx_test = get_idx(y_groups_test)

    if dataset in ("compas", "adult", "bank"):
        model = MLP(
            feature_size=X_train.shape[1],
            hidden_dim=50,
            num_classes=2,
            num_layer=2
        )
    elif dataset in ("UTKFace_preprocessed"):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

    optimizer = SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction='none')

    epoch = 20
    iteration = 30

    multipliers = np.zeros(len(protected_train)) if (fairness_constraint == 'eopp' or fairness_constraint == 'dp') else np.zeros(len(protected_train) * 2)
    eta = 3.

    max_iter = 0
    max_tradeoff = 0

    naive_acc = 0
    naive_vio = 0

    scale_factor = 50

    for _iter in range(iteration):
        print("Iteration: {}".format(_iter))
        if _iter == 0 or method == 'naive': weights = torch.ones(len(X_train))
        elif method == 'influence' and _iter >= 1:
            start = time()
            weights = torch.tensor(calc_influence_dataset(X_train, y_train, constraint_idx_train, X_groups_train, y_groups_train,
                                                            model, train_loader, gpu=gpu, constraint=fairness_constraint, r=1))
            end = time()
            print("Elapsed time for calculating weights {}".format(end-start))
            weights = exp_normalize(weights, scale_factor)
        elif method == 'reweighting' and _iter >= 1:
            weights = torch.tensor(debias_weights(fairness_constraint, y_train, protected_train, multipliers))

        print("Weights: {}".format(weights))
        for _epoch in tqdm(range(epoch)):
            model.train()
            i = 0
            for z, t in train_loader:
                if gpu >= 0: z, t, model = z.cuda(), t.cuda(), model.cuda()
                model.train()
                y_pred = model(z)

                weight = weights[i * batch_size: (i + 1) * batch_size] if (i + 1) * batch_size <= len(X_train) else weights[i * batch_size:]
                if gpu >= 0: weight = weight.cuda()

                loss = torch.mean(weight * criterion(y_pred, t))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1

        if method == 'reweighting':
            y_pred_train = model(X_train).argmax(dim=1).detach().numpy()
            _, violations = get_error_and_violations(fairness_constraint, y_pred_train, y_train, protected_train)
            multipliers += eta * np.array(violations)

        model.eval()
        with torch.no_grad():
            if gpu >= 0: X_test, y_test = X_test.cuda(), y_test.cuda()
            accuracy = sum(model(X_test).argmax(dim=1) == y_test) / len(y_test)

        print("Iteration {}, Test Acc: {}".format(_iter, accuracy))

        violation = calc_loss_diff(fairness_constraint, X_groups_test, y_groups_test, constraint_idx_test, model)
        print("Iteration {}, Violation: {}".format(_iter, violation))

        fairness_metric = calc_fairness_metric(fairness_constraint, X_groups_test, y_groups_test, model)
        print("Iteration {}, Fairness metric: {}%".format(_iter, fairness_metric * 100))


    if method == 'naive':
        influence_scores = np.array(calc_influence_dataset(X_train, y_train, constraint_idx_train, X_groups_train, y_groups_train,
                                                            model, train_loader, gpu=gpu, constraint=fairness_constraint))
        top_10_idx = np.argpartition(influence_scores, -10)[-10:]
        print(top_10_idx)
        print(X_train[top_10_idx])


if __name__ == '__main__':
    main()



