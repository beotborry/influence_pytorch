import torch
from dataloader import get_data, CustomDataset
from utils import split_dataset
import numpy as np
from mlp import MLP
from torch.optim import SGD
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix



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

iteration = 50
epochs = 20
max_iter = 0
max_tradeoff = 0

multipliers = np.zeros(len(protected_train))
weights = np.array([1] * X_train.shape[0])
eta = 1.

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

    if (accuracy / i) / eopp_metrics > max_tradeoff:
        max_tradeoff = (accuracy / i) / eopp_metrics
        max_iter = iter
        torch.save(model, "./model/reweighting2")

print(max_iter, max_tradeoff)


