from dataloader_factory import DataloaderFactory
import torchvision.models as models
from torch.optim import SGD, Adam
import torch.nn as nn
from utils import get_accuracy
from tqdm import tqdm
import torch
import numpy as np
from utils import get_eopp_idx
from influence_function import calc_influence_dataset, calc_influence_dataset_with_dataset
from dataset_factory import DatasetFactory
from torchvision import transforms
from utils import split_dataset
from torch.utils.data import DataLoader


def _init_fn(worker_id):
    np.random.seed(int(0))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_list = [transforms.RandomResizedCrop(176),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize
                  ]

preprocessing = transforms.Compose(transform_list)

train_dataset = DatasetFactory.get_dataset("celeba", preprocessing, split='train', target='Attractive',
                                           seed=0, skew_ratio=1., labelwise=False)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                          num_workers=0, worker_init_fn=_init_fn, pin_memory=True,
                          drop_last=True)

X_train = []
y_train = train_dataset.attr[:,train_dataset.target_idx]
protected_train = train_dataset.attr[:, train_dataset.sensi_idx]

X_groups_train = [[],[]]
y_groups_train = [[],[]]
for idx, sen_val in enumerate(protected_train):
    X_groups_train[sen_val].append(idx)
    y_groups_train[sen_val].append(y_train[idx])

X_groups_train[0] = torch.LongTensor(X_groups_train[0])
X_groups_train[1] = torch.LongTensor(X_groups_train[1])
y_groups_train[0] = torch.LongTensor(y_groups_train[0])
y_groups_train[1] = torch.LongTensor(y_groups_train[1])


constraint_idx_train = get_eopp_idx(y_groups_train)


model = torch.load("./model/celeba_cpu", map_location=torch.device('cpu'))


influence_scores = np.array(
    calc_influence_dataset_with_dataset(train_dataset, y_train, constraint_idx_train, X_groups_train, y_groups_train, model, train_loader, gpu=-1, constraint="eopp"))

top_10_idx = np.argpartition(influence_scores, -10)[-10:]
print(top_10_idx)
print(X_train[top_10_idx])

