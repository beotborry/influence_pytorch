import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from dataloader import get_data, CustomDataset
from torch.utils.data import DataLoader
from utils import split_dataset


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

model = torch.load('./model/influence2')
model.eval()

y_female_eopp = y_female_test[female_1_idx_test].detach().numpy()
y_female_eopp_pred = model(X_female_test[female_1_idx_test]).argmax(dim=1).detach().numpy()

print(y_female_test)
female_confusion_mat = confusion_matrix(y_female_test.detach().numpy(), model(X_female_test).argmax(dim=1).detach().numpy()).ravel()
male_confusion_mat = confusion_matrix(y_male_test.detach().numpy(), model(X_male_test).argmax(dim=1).detach().numpy()).ravel()

print(female_confusion_mat, male_confusion_mat)

_, _, f_fn, f_tp = female_confusion_mat
_, _, m_fn, m_tp = male_confusion_mat

print("Eopp violation: {}".format(f_tp/(f_fn+f_tp) - m_tp/(m_fn+m_tp)))

batch_size = 128
test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

gpu = 1 if torch.cuda.is_available() else -1


y_pred_test = model(X_test).argmax(dim=1)
accuracy = sum(y_pred_test == y_test) / len(y_test)
print("Acc: {}".format(accuracy))
