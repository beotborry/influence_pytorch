from adult_dataloader import get_data
import torch
from torch.optim import SGD, Adam
from utils import split_dataset, exp_normalize
from mlp import MLP
from influence_function import s_test, avg_s_test
from tqdm import tqdm
import torch.nn as nn
from influence_function import calc_influence
from adult_dataloader import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import time


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, protected_train, protected_test = get_data()


    X_female_train, y_female_train = split_dataset(X_train, y_train, protected_train[0])
    X_male_train, y_male_train = split_dataset(X_train, y_train, protected_train[1])
    X_male_train = torch.FloatTensor(X_male_train)
    y_male_train = torch.LongTensor(y_male_train)
    X_female_train = torch.FloatTensor(X_female_train)
    y_female_train = torch.LongTensor(y_female_train)

    female_1_idx_train = np.where(y_female_train.numpy() == 1)
    male_1_idx_train = np.where(y_male_train.numpy() == 1)

    X_female_test, y_female_test = split_dataset(X_test, y_test, protected_test[0])
    X_male_test, y_male_test = split_dataset(X_test, y_test, protected_test[1])
    X_male_test = torch.FloatTensor(X_male_test)
    y_male_test = torch.LongTensor(y_male_test)
    X_female_test = torch.FloatTensor(X_female_test)
    y_female_test = torch.LongTensor(y_female_test)

    female_1_idx_test = np.where(y_female_test.numpy() == 1)
    male_1_idx_test = np.where(y_male_test.numpy() == 1)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    num_features = 122

    model = MLP(
        feature_size=122,
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

def calc_influence_dataset(X, y, X_female_train, y_female_train, X_male_train, y_male_train,
                           model, train_loader, gpu):

    s_test_vec = avg_s_test(z_group1=X_female_train[female_1_idx_train], t_group1=y_female_train[female_1_idx_train],
                            z_group2=X_male_train[male_1_idx_train], t_group2=y_male_train[male_1_idx_train],
                            model = model, z_loader=train_loader, gpu=gpu, r=7)
    #s_test_vec = s_test(z_group1=X_female_train[female_1_idx_train], t_group1=y_female_train[female_1_idx_train],
    #                    z_group2=X_male_train[male_1_idx_train], t_group2=y_male_train[male_1_idx_train],
    #                    model=model, z_loader=train_loader, gpu=gpu)

    influences = []
    for z, t in zip(X, y):
        influences.append(calc_influence(z, t, s_test_vec, model, train_loader, gpu=gpu))
    return influences

iteration = 30

max_iter = 0
max_tradeoff = 0
scale_factor = 150

start = time.time()
for iter in range(iteration):
    print("Iteration: {}".format(iter))
    if iter == 0: weights = torch.ones(len(X_train))
    else:
        weights = torch.tensor(calc_influence_dataset(X_train, y_train, X_female_train, y_female_train,
                                                      X_male_train, y_male_train,
                                                      model, train_loader, gpu = gpu))
    weights = exp_normalize(weights, scale_factor)
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

    if gpu >= 0: X_female_train, y_female_train, X_male_train, y_male_train = X_female_train.cuda(), y_female_train.cuda(), X_male_train.cuda(), y_male_train.cuda()

    female_loss = nn.CrossEntropyLoss()(model(X_female_test[female_1_idx_test]), y_female_test[female_1_idx_test])
    print("Iteration {}, Female Loss: {}".format(iter, female_loss))

    male_loss = nn.CrossEntropyLoss()(model(X_male_test[male_1_idx_test]), y_male_test[male_1_idx_test])
    print("Iteration {}, Male Loss: {}".format(iter, male_loss))

    violation = abs(male_loss - female_loss)
    print("Iteration {}, Violation: {}".format(iter, violation))

    female_confusion_mat = confusion_matrix(y_female_test.detach().numpy(),
                                            model(X_female_test).argmax(dim=1).detach().numpy()).ravel()
    male_confusion_mat = confusion_matrix(y_male_test.detach().numpy(),
                                          model(X_male_test).argmax(dim=1).detach().numpy()).ravel()

    _, _, f_fn, f_tp = female_confusion_mat
    _, _, m_fn, m_tp = male_confusion_mat

    eopp_metrics = abs(f_tp / (f_fn + f_tp) - m_tp / (m_fn + m_tp))
    print("Eopp metrics: {}".format(abs(eopp_metrics)))
    #print(female_confusion_mat, male_confusion_mat)

    if (accuracy / i) / eopp_metrics > max_tradeoff and iter >= 2:
        max_tradeoff = (accuracy / i) / eopp_metrics
        max_iter = iter
        #torch.save(model, "./model/influence4")

print(max_iter, max_tradeoff)
end = time.time()
print((end - start) / 60)


