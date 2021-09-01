from dataloader import get_data, CustomDataset
import torch.nn as nn
import torch
from torch.optim import SGD
from model import LinearModel
import pytorch_influence_functions as ptif
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import numpy as np
from utils import split_dataset, exp_normalize
from mlp import MLP
from torch.autograd import grad
from influence_function import grad_z, s_test, calc_influence
import time
from tqdm import tqdm


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, protected_train, protected_test = get_data()


    X_female_train, y_female_train = split_dataset(X_train, y_train, protected_train[0], "EOpp")
    X_male_train, y_male_train = split_dataset(X_train, y_train, protected_train[1], "EOpp")
    X_male_train = torch.FloatTensor(X_male_train)
    y_male_train = torch.LongTensor(y_male_train)
    X_female_train = torch.FloatTensor(X_female_train)
    y_female_train = torch.LongTensor(y_female_train)

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

    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(reduction = 'none')

from dataloader import CustomDataset
from torch.utils.data import DataLoader

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)

test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

gpu = 1 if torch.cuda.is_available() else -1
print(gpu)

from influence_function import calc_influence
import time

def calc_influence_dataset(X, y,  X_female_train, y_female_train, X_male_train, y_male_train,
                     model, train_loader):

  s_test_vec = s_test(z_group1=X_female_train, t_group1=y_female_train, z_group2=X_male_train, t_group2=y_male_train,
                      model=model, z_loader=train_loader)
  influences = []
  for z, t in zip(X, y):
      influences.append(calc_influence(z, t, s_test_vec, model, train_loader))

  return influences


for epoch in range(10):
    print("Epoch: {}".format(epoch))
    for z, t in tqdm(train_loader):
        if gpu >= 0: z, t, model = z.cuda(), t.cuda(), model.cuda()
        model.train()
        y_pred = model(z)
        start = 0
        if epoch == 0:
            weights = torch.ones(len(t))
        else:
            weights = torch.tensor(calc_influence_dataset(z, t,
                                                          X_female_train, y_female_train,
                                                          X_male_train, y_male_train,
                                                          model, train_loader))
            # add normalization of influence scores
            weights = exp_normalize(weights)
        if gpu >= 0: weights = weights.cuda()

        loss = torch.mean(weights * criterion(y_pred, t))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    print()
    with torch.no_grad():
        for z, t in test_loader:
            if gpu >= 0: z, t = z.cuda(), t.cuda()
            y_pred = model(z)
            accuracy = (((y_pred.argmax(dim=1) == t).sum()) / len(t)).item()
            print("Epoch {}, Acc: {}".format(epoch, accuracy))
            break

    if gpu >= 0: X_female_train, y_female_train, X_male_train, y_male_train = X_female_train.cuda(), y_female_train.cuda(), X_male_train.cuda(), y_male_train.cuda()

    female_loss = nn.CrossEntropyLoss()(model(X_female_train), y_female_train)
    print("Epoch {}, Female Loss: {}".format(epoch, female_loss))

    male_loss = nn.CrossEntropyLoss()(model(X_male_train), y_male_train)
    print("Epoch {}, Male Loss: {}".format(epoch, male_loss))

    violation = abs(male_loss - female_loss)
    print("Epoch {}, Violation: {}".format(epoch, violation))



'''


    model = LinearModel(num_features, num_classes=2)
    optimizer = SGD(model.parameters(), lr=0.03)
    loss_fn = nn.BCELoss()

    # Split dataset
    X_female_train, y_female_train = split_dataset(X_train, y_train, protected_train[0], "EOpp")
    X_male_train, y_male_train = split_dataset(X_train, y_train, protected_train[1], "EOpp")

    # Array to Tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    X_female_train = torch.FloatTensor(X_female_train)
    y_female_train = torch.FloatTensor(y_female_train)
    X_male_train = torch.FloatTensor(X_male_train)
    y_male_train = torch.FloatTensor(y_male_train)

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Training
    for epoch in range(1, 500):
        model.train()
        for inputs, target in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = loss_fn(y_pred.squeeze(), y_train)
            loss.backward()
            optimizer.step()

        female_loss = loss_fn(model(X_female_train).squeeze(), y_female_train)
        male_loss = loss_fn(model(X_male_train).squeeze(), y_male_train)
        loss_diff = abs(female_loss.item() - male_loss.item())

        print('Epoch ', epoch, ':')
        print('female loss is {}'.format(female_loss.item()))
        print('male loss is {}'.format(male_loss.item()))
        print('difference is {}'.format(loss_diff))

        model.eval()
        correct = 0
        test_loss = loss_fn(model(X_test).squeeze(), y_test)

        print('test loss is {}'.format(test_loss.item()))
        correct += ((model(X_test) == y_test).sum().item())
        print('Accuracy of the network: %d %%' % (100 * correct / len(y_test)))
'''



