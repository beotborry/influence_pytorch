import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from compas_dataloader import get_data
from adult_dataloader import CustomDataset
from torch.utils.data import DataLoader
from utils import split_dataset_multi, get_eopp_idx

X_train, y_train, X_test, y_test, protected_train, protected_test = get_data()

z_groups_train, t_groups_train = split_dataset_multi(X_train, y_train, protected_train)
z_groups_test, t_groups_test = split_dataset_multi(X_test, y_test, protected_test)

eopp_idx_train = get_eopp_idx(t_groups_train)
eopp_idx_test = get_eopp_idx(t_groups_test)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

model = torch.load('./model/compas_influence_best')
model.eval()

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

batch_size = 128
test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

gpu = 1 if torch.cuda.is_available() else -1

y_pred_test = model(X_test).argmax(dim=1)
accuracy = sum(y_pred_test == y_test) / len(y_test)
print("Acc: {}".format(accuracy))
