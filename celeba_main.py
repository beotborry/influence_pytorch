from dataloader_factory import DataloaderFactory
import torchvision.models as models
from torch.optim import SGD, Adam
import torch.nn as nn
from utils import get_accuracy
from tqdm import tqdm
import torch
import numpy as np
from utils import get_eopp_idx
from influence_function import calc_influence_dataset
import gc

tmp = DataloaderFactory.get_dataloader("celeba", img_size=176,
                                                    batch_size=128,
                                                    num_workers=0,
                                                    target='Attractive')

num_classes, num_groups, train_loader, test_loader = tmp

model = models.resnet18(pretrained=True)

optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 5

gpu = False

model.train()
for epoch in range(epochs):
    running_acc = 0.0
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, _, groups, targets, _ = data
        labels = targets

        if gpu: inputs, labels, model = inputs.cuda(), labels.cuda(), model.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        running_acc += get_accuracy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: {}, Trng Acc: {}, Trng loss: {}".format(epoch, running_acc / i, running_loss / i))
    # Evaluation
    running_acc_test = 0.0
    running_loss_test = 0.0
    for i, data in enumerate(test_loader):
        inputs, _, groups, targets, _ = data
        labels = targets

        if gpu: inputs, labels, model = inputs.cuda(), labels.cuda(), model.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss_test += loss.item()
        running_acc_test += get_accuracy(outputs, labels)
    print("Test Acc: {}, Test loss: {}".format(running_acc_test / i, running_loss_test / i))

torch.save(model, './model/celeba_cpu')

